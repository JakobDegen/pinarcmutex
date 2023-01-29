//! This crate provides the [`PinArcMutex`] type, which offers shared mutable access to pinned
//! state.
//!
//! ### Problem
//!
//! Rust already provides good solutions to the problem of shared mutable state - often, this looks
//! like `Arc<Mutex<T>>`. However, sometimes we additionally need that shared mutable state to be
//! pinned. One example of this is wanting to poll a [`Stream`] from multiple threads.
//!
//! [`Stream`]: https://docs.rs/futures/latest/futures/stream/trait.Stream.html
//!
//! It turns out that there are no great ways to do this. The fundamental problem is that `Mutex`
//! types in std and tokio have no [structural pinning]. Unfortunately, that cannot be solved by
//! designing a better `PinnedMutex` type either - the real limitation is actually in [`Pin`]
//! itself. Specifically, because of methods like [`get_ref`], the `Pin` API makes it impossible to
//! assert pinned-ness of a `T` without creating a `&mut T`, which is prohibitive for cases like
//! mutexes.
//!
//! [`get_ref`]: std::pin::Pin::get_ref
//! [structural pinning]:
//!     https://doc.rust-lang.org/std/pin/index.html#projections-and-structural-pinning
//!
//! ### Alternatives
//!
//! If your `T` is `Unpin`, you can use a `Arc<tokio::Mutex<T>>` directly and do not need this
//! crate.
//!
//! If you do not mind an extra allocation, you can also get a similar API without an extra
//! dependency via `Arc<tokio::Mutex<Pin<Box<T>>>>`.
//!
//! ### MSRV
//!
//! This crate has the same MSRV as its only dependency, tokio.
//!

use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::sync::MutexGuard as TokioGuard;

/// A type that resembles an `Arc<Mutex<T>>`, expected that the backing data is pinned.
///
/// The API surface of this type is essentially the combined API surface of [`Arc`] and
/// [`tokio::sync::Mutex`]. Specifically, there is an `Arc`-like [`Clone`] impl and a `Mutex`-like
/// [`lock`](PinArcMutex::lock) impl. The only difference is that the backing data is pinned - this
/// means that locking the mutex gets you access to a `Pin<&mut T>` instead of a `&mut T`.
#[derive(Clone, Debug, Default)]
pub struct PinArcMutex<T> {
    // Safety invariant: The data inside the `Mutex` is pinned. All public APIs that make a pointer
    // to this data available must include a safety comment documenting why that API conforms to the
    // `Pin` requirements.
    inner: Arc<Mutex<T>>,
}

impl<T> PinArcMutex<T> {
    /// Creates a new `PinArcMutex<T>` storing the given value.
    pub fn new(val: T) -> PinArcMutex<T> {
        PinArcMutex {
            inner: Arc::new(Mutex::new(val)),
        }
    }

    /// Locks the inner mutex, returning a `MutexGuard`.
    ///
    /// This works almost exactly like the [`tokio::sync::Mutex::lock`] method, except that the
    /// `MutexGuard` that is returned hands out `Pin<&mut T>`s instead of `&mut T`s, via the
    /// [`MutexGuard::get`] method.
    ///
    /// The returned `MutexGuard` implements [`Deref`](std::ops::Deref) as would be expected.
    /// However, because [`DerefMut`](std::ops::DerefMut) must return a `&mut T`, that is only
    /// implemented when `T: Unpin`.
    pub async fn lock(&self) -> MutexGuard<'_, T> {
        MutexGuard {
            inner: self.inner.lock().await,
        }
    }

    /// Gets a mutable reference to the underlying data, if there are no other `PinArcMutex`s
    /// pointing to the same data.
    ///
    /// This is a combination of the [`Arc::get_mut`] and [`tokio::sync::Mutex::get_mut`] methods.
    pub fn get_mut(&mut self) -> Option<Pin<&mut T>> {
        // SAFETY: The data is pinned
        unsafe { Some(Pin::new_unchecked(Arc::get_mut(&mut self.inner)?.get_mut())) }
    }

    /// Returns true if two `PinArcMutex`s point to the same data.
    ///
    /// Analogous to [`Arc::ptr_eq`].
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        Arc::ptr_eq(&this.inner, &other.inner)
    }
}

/// The RAII type that is returned from locking a [`PinArcMutex`].
///
/// See the [`PinArcMutex::lock`] method for full details.
pub struct MutexGuard<'a, T> {
    inner: TokioGuard<'a, T>,
}

impl<'a, T> MutexGuard<'a, T> {
    /// Returns a pinned mutable reference to the locked data.
    pub fn get(&mut self) -> Pin<&mut T> {
        // SAFETY: The data inside the `Mutex` is pinned
        unsafe { Pin::new_unchecked(&mut *self.inner) }
    }
}

impl<'a, T> std::ops::Deref for MutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: The `Pin` contract does not prohibit creating a `&T` to pinned data.
        &*self.inner
    }
}

impl<'a, T> std::ops::DerefMut for MutexGuard<'a, T>
where
    T: Unpin,
{
    fn deref_mut(&mut self) -> &mut T {
        self.get().get_mut()
    }
}
