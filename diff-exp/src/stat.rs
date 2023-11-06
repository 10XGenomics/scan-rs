//! statistics
//!
//! ## `Statistics` trait
//!
//! * To make generic code, there is `Statistics` trait
//!     * `mean`: just mean

//!     ```rust
//!     pub trait Statistics {
//!         type Array;
//!         type Value;
//!
//!         fn mean(&self) -> Self::Value;
//!         fn sum(&self) -> Self::Value;
//!         fn var(&self, ddof:f64) -> Self::Value;
//!         fn percentile(&self) -> Self::Value;
//!         fn median(&self) -> Self::Value;
//!     }
//!     ```
//!

use num_traits::Float;
use std::cmp::Ordering::{self, Equal, Greater, Less};
use std::mem;
/// Statistics Trait
///
/// It contains `mean`, `sum`, `var`, `percentile`, `median`

pub trait Statistics {
    /// array is the type of the container
    //type Array;
    /// value is the type of the Value
    type Value: Float;
    /// Sum
    fn sum(&self) -> Self::Value;
    /// mean stats
    fn mean(&self) -> Self::Value;
    /// var stats
    fn var(&self, ddof: f64) -> Self::Value;
    /// Percentile: the value below which `pct` percent of the values in `self` fall. For example,
    fn percentile(&self, pct: f64) -> Self::Value;
    /// medium
    fn median(&self) -> Self::Value;
}

impl<T: std::clone::Clone + num_traits::cast::ToPrimitive + PartialOrd + Copy> Statistics for [T] {
    type Value = f64;

    fn sum(&self) -> f64 {
        let mut partials = vec![];

        for &x in self {
            let mut x = T::to_f64(&x).unwrap();
            let mut j = 0;
            // This inner loop applies `hi`/`lo` summation to each
            // partial so that the list of partial sums remains exact.
            for i in 0..partials.len() {
                let mut y: f64 = partials[i];
                if x.abs() < y.abs() {
                    mem::swap(&mut x, &mut y);
                }
                // Rounded `x+y` is stored in `hi` with round-off stored in
                // `lo`. Together `hi+lo` are exactly equal to `x+y`.
                let hi = x + y;
                let lo = y - (hi - x);
                if lo != 0.0 {
                    partials[j] = lo;
                    j += 1;
                }
                x = hi;
            }
            if j >= partials.len() {
                partials.push(x);
            } else {
                partials[j] = x;
                partials.truncate(j + 1);
            }
        }
        let zero: f64 = 0.0;
        partials.iter().fold(zero, |p, q| p + *q)
    }

    fn mean(&self) -> f64 {
        assert!(!self.is_empty());
        self.sum() / (self.len() as f64)
    }

    fn var(&self, ddof: f64) -> f64 {
        if (self.len() as f64) < ddof {
            0.0
        } else {
            let mean = self.mean();
            let mut v: f64 = 0.0;
            for s in self {
                let ss = T::to_f64(s).unwrap();
                let x = ss - mean;
                v += x * x;
            }
            // N.B., this is _supposed to be_ len-1, not len. If you
            // change it back to len, you will be calculating a
            // population variance, not a sample variance.
            let denom = (self.len() as f64) - ddof;
            v / denom
        }
    }

    fn percentile(&self, pct: f64) -> f64 {
        let mut tmp = Vec::<T>::new();
        for s in self {
            tmp.push(*s);
        }
        local_sort(&mut tmp);
        percentile_of_sorted(tmp.as_slice(), pct)
    }

    fn median(&self) -> f64 {
        self.percentile(50_f64)
    }
}

fn local_sort<T: PartialOrd + Copy>(v: &mut [T]) {
    v.sort_by(|x: &T, y: &T| local_cmp(*x, *y));
}

fn local_cmp<T>(x: T, y: T) -> Ordering
where
    T: PartialOrd,
{
    if x < y {
        Less
    } else if x == y {
        Equal
    } else {
        Greater
    }
}

// Helper function: extract a value representing the `pct` percentile of a sorted sample-set, using
// linear interpolation. If samples are not sorted, return nonsensical value.
fn percentile_of_sorted<T>(sorted_samples: &[T], pct: f64) -> f64
where
    T: num_traits::cast::ToPrimitive,
{
    assert!(!sorted_samples.is_empty());
    if sorted_samples.len() == 1 {
        return T::to_f64(&sorted_samples[0]).unwrap();
    }
    let zero: f64 = 0.0;
    assert!(zero <= pct);
    let hundred = 100_f64;
    assert!(pct <= hundred);
    if pct == hundred {
        return T::to_f64(&sorted_samples[sorted_samples.len() - 1]).unwrap();
    }
    let length = (sorted_samples.len() - 1) as f64;
    let rank = (pct / hundred) * length;
    let l_rank = rank.floor();
    let d = rank - l_rank;
    let n = l_rank as usize;
    let lo = T::to_f64(&sorted_samples[n]).unwrap();
    let hi = T::to_f64(&sorted_samples[n + 1]).unwrap();
    lo + (hi - lo) * d
}

#[cfg(test)]
mod test {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_stats() {
        let v = vec![1, 2, 4, 3, 5, 6];
        assert_approx_eq!(v.mean(), 3.5, 1e-11);
        assert_approx_eq!(v.var(0.0), 2.9166666666666665, 1e-11);
        assert_approx_eq!(v.median(), 3.5, 1e-11);
        assert_approx_eq!(v.percentile(0.95), 1.0475, 1e-11);

        let mut v = vec![1.0f64];
        v.append(&mut vec![1e-12f64; 1e6 as usize]);
        assert_approx_eq!(v.mean(), 9.999_999_999_999_974e-7, 1e-13);
        assert_approx_eq!(v.var(0.0), 9.999_980_000_010_034e-7, 1e-13);
        assert_approx_eq!(v.median(), 1e-12, 1e-13);
        assert_approx_eq!(v.percentile(0.95 * 100.0), 1e-12, 1e-13);
    }
}
