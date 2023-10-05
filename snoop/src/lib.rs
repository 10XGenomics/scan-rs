use std::fmt::Display;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Debug)]
pub struct CancellationError;

impl std::error::Error for CancellationError {
    fn description(&self) -> &str {
        "computation was cancelled"
    }
}

impl Display for CancellationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancellation error")
    }
}

pub trait Cancel: Send + Sync {
    /// true if the snoop is cancelled, false if not.
    fn is_cancelled(&self) -> bool;

    /// creates a new snoop that shares the same cancel state
    fn get_cancel_subsnoop(&self) -> Self;
}

pub trait Progress: Send + Sync {
    /// sets the fractional progress. fraction should be [0.0, 1.0]
    fn set_progress(&mut self, fraction: f64);

    /// creates a new snoop whose progress is limited to the fraction passed.
    /// fraction should be between [0.0, 1.0] and also not greater than the amount of progress remaining.
    fn get_progress_subsnoop(&mut self, fraction: f64) -> Self;
}

pub trait CancelProgress: Cancel + Progress + Send + Sync {
    /// creates a new snoop whose progress is limited to the fraction passed.
    /// This new snoop shares the same cancel state.
    /// fraction should be between [0.0, 1.0] and also not greater than the amount of progress remaining.
    fn get_subsnoop(&mut self, fraction: f64) -> Self;

    /// sets the fractional progress. fraction should be [0.0, 1.0]
    /// also checks if the snoop is cancelled and if so returns an Err
    fn set_progress_check(&mut self, progress: f64) -> Result<(), CancellationError> {
        // it _might_ be useful for debugging if this error captured it's
        // stack trace, so then you could find out where the algo was when it cancelled,
        // which might be useful for debugging "stalled" computations, but it's
        // probably not worth having to pull in all the backtrace machinery here.
        // (consider adding this once backtrace is stable)
        if self.is_cancelled() {
            return Err(CancellationError);
        }

        self.set_progress(progress);
        Ok(())
    }
}

#[derive(Default, Copy, Clone)]
pub struct NoOpSnoop;

impl Cancel for NoOpSnoop {
    fn is_cancelled(&self) -> bool {
        false
    }

    fn get_cancel_subsnoop(&self) -> Self {
        *self
    }
}

impl Progress for NoOpSnoop {
    fn set_progress(&mut self, _frac: f64) {}

    fn get_progress_subsnoop(&mut self, _fraction: f64) -> Self {
        *self
    }
}

impl CancelProgress for NoOpSnoop {
    fn get_subsnoop(&mut self, _fraction: f64) -> Self {
        *self
    }
}

#[derive(Debug, Default)]
pub struct AtomicState {
    cancelled: AtomicBool,
    progress: AtomicU64,
}

type AtomicProgress = u64;

impl AtomicState {
    /// Shift max progress to account for float rounding errors that could lead to overflow
    const MAX_PROGRESS: AtomicProgress = AtomicProgress::MAX >> 1;

    pub fn new() -> Self {
        Default::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    pub fn get_progress(&self) -> f64 {
        let progress = self.progress.load(Ordering::Relaxed);
        progress as f64 / AtomicState::MAX_PROGRESS as f64
    }
}

#[derive(Debug)]
pub struct AtomicSnoop {
    state: Arc<AtomicState>,
    progress: AtomicProgress,
    total_progress: AtomicProgress,
}

pub fn atomic() -> (Arc<AtomicState>, AtomicSnoop) {
    let state: Arc<AtomicState> = Default::default();
    (state.clone(), AtomicSnoop::new(state))
}

impl AtomicSnoop {
    fn new(state: Arc<AtomicState>) -> Self {
        Self {
            total_progress: AtomicState::MAX_PROGRESS,
            progress: 0,
            state,
        }
    }

    pub fn cancel(&self) {
        self.state.cancel()
    }

    /// returns progress as a fraction [0, 1.0]
    fn fraction(&self) -> f64 {
        self.progress as f64 / self.total_progress as f64
    }

    fn fraction_to_progress(&self, fraction: f64) -> AtomicProgress {
        (fraction * self.total_progress as f64) as AtomicProgress
    }
}

impl Cancel for AtomicSnoop {
    fn is_cancelled(&self) -> bool {
        self.state.is_cancelled()
    }

    fn get_cancel_subsnoop(&self) -> Self {
        AtomicSnoop {
            total_progress: 0, // this disables any progress ever being set
            progress: 0,
            state: self.state.clone(),
        }
    }
}

impl Progress for AtomicSnoop {
    /// Calculates change in progress and atomically adds this to the state
    fn set_progress(&mut self, fraction: f64) {
        debug_assert!(fraction >= 0.0);
        debug_assert!(fraction <= 1.0);

        let next_progress = self.fraction_to_progress(fraction);

        if next_progress > self.progress {
            let delta = next_progress - self.progress;
            self.state.progress.fetch_add(delta, Ordering::Relaxed);
        } else {
            let delta = self.progress - next_progress;
            self.state.progress.fetch_sub(delta, Ordering::Relaxed);
        }

        self.progress = next_progress;
    }

    fn get_progress_subsnoop(&mut self, fraction: f64) -> AtomicSnoop {
        debug_assert!(fraction >= 0.0);
        debug_assert!(fraction <= 1.0);
        debug_assert!(fraction + self.fraction() <= 1.0);

        let subsnoop_max_progress = self.fraction_to_progress(fraction);
        let next_progress = self.progress + subsnoop_max_progress;

        // Here we either need to update this snoop's max_progress or update
        // it's current progress to avoid doubly reporting progress from the subsnoop
        // and this snoop.
        //
        // For example, with no changes this code would end up reporting 1.5 total progress:
        //   snoop.set_progress(0.5);
        //   let subsnoop = snoop.get_subsnoop(0.5)
        //   subsnoop.set_progress(1.0);
        //   snoop.set_progress(1.0);
        //
        // If we update the `max_progress`, but not the current progress then the above code still
        // fails.  When `snoop.set_progress(1.0)` is called it will calculate a delta change of
        // 0.5 over the new max_progress. This will lead to a total progress of 1.25.
        //
        // Instead the solution is to just update the snoop's `progress` as if it has progressed
        // the amount of the subsnoop.  This does mean that we can't calculate an actual
        // progress for a snoop and can only rely on the progress of the top level AtomicState.
        // To solve this we would probably have to use lifetimes and have children snoop hold
        // references to parents.
        self.progress = next_progress;

        AtomicSnoop {
            total_progress: subsnoop_max_progress,
            progress: 0,
            state: self.state.clone(),
        }
    }
}

impl CancelProgress for AtomicSnoop {
    fn get_subsnoop(&mut self, fraction: f64) -> Self {
        self.get_progress_subsnoop(fraction)
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn test_atomic() {
        let (state, mut snoop) = atomic();

        // simple progress
        assert_eq!(state.get_progress(), 0.0);
        snoop.set_progress(0.25);
        assert_eq!(state.get_progress(), 0.25);
        snoop.set_progress(1.0);
        assert_eq!(state.get_progress(), 1.0);

        // simple cancellation
        assert!(!state.cancelled.load(Ordering::Relaxed));
        assert!(!snoop.is_cancelled());
        state.cancel();
        assert!(state.cancelled.load(Ordering::Relaxed));
        assert!(snoop.is_cancelled());
    }

    #[test]
    fn test_atomic_subsnoop() {
        let (state, mut snoop) = atomic();

        let mut snoop1 = snoop.get_subsnoop(0.25);
        snoop1.set_progress(1.0);
        assert_eq!(state.get_progress(), 0.25);

        let mut snoop2 = snoop.get_subsnoop(0.25);
        snoop2.set_progress(1.0);
        assert_eq!(state.get_progress(), 0.50);

        let mut snoop3 = snoop.get_subsnoop(0.25);
        snoop3.set_progress(1.0);
        assert_eq!(state.get_progress(), 0.75);

        let mut snoop4 = snoop.get_subsnoop(0.25);
        snoop4.set_progress(1.0);
        assert_eq!(state.get_progress(), 1.0);

        // subsnoop cancellation
        state.cancel();
        assert!(snoop.is_cancelled());
        assert!(snoop1.is_cancelled());
        assert!(snoop2.is_cancelled());
        assert!(snoop3.is_cancelled());
        assert!(snoop4.is_cancelled());
    }
}
