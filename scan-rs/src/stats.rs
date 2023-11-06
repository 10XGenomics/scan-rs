//! Statistics functions

use ndarray::prelude::*;
use ndarray::DataMut;
use ndarray_stats::errors::QuantileError;
use ndarray_stats::interpolate::Midpoint;
use ndarray_stats::Quantile1dExt;
use noisy_float::prelude::n64;
use num_traits::FromPrimitive;
use std::ops::{Add, Div, Mul, Rem, Sub};

/// Return the median. Sorts its argument in place.
pub fn median_mut<S, T>(xs: &mut ArrayBase<S, Ix1>) -> Result<T, QuantileError>
where
    S: DataMut<Elem = T>,
    T: Clone + Copy + Ord + FromPrimitive,
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Rem<Output = T>,
{
    if false {
        // quantile_mut may fail with the error: fatal runtime error: stack overflow
        // See CELLRANGER-5331 and https://github.com/rust-ndarray/ndarray-stats/issues/86
        xs.quantile_mut(n64(0.5), &Midpoint)
    } else {
        if xs.is_empty() {
            return Err(QuantileError::EmptyInput);
        }
        let slice = xs.as_slice_mut();
        match slice {
            Some(vector) => vector.sort_unstable(),
            None => panic!("An attempt was made to calculate a median value for non-contiguous data"),
        }
        Ok(if xs.len() % 2 == 0 {
            (xs[xs.len() / 2] + xs[xs.len() / 2 - 1]) / (T::from_u64(2).unwrap())
        } else {
            xs[xs.len() / 2]
        })
    }
}

// A bespoke median calculation method to get around a stack_overflow issue in ndarray_stats, see
// comments in `median_mut` for links to descriptions of this issue.
// For a given 2D matrix, this method returns the median along rows.  This method is constrained to
// only calculate along rows and not columns because by default a Array2 has rows contiguous in
// memory, which allows them to return a Some(_) from `.as_slice_mut` that can be passed to
// `sort_unstable` for easy calculation of the median, and it's also a more efficient access
// pattern.
pub(crate) fn median_array_rows_mut<S, T>(x: &mut ArrayBase<S, Ix2>) -> Result<Vec<T>, QuantileError>
where
    S: DataMut<Elem = T>,
    T: Clone + Copy + Ord + FromPrimitive,
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Rem<Output = T>,
{
    let result: Result<Vec<_>, _> = x
        .axis_iter_mut(Axis(0))
        .map(|mut f: ArrayViewMut<T, Ix1>| median_mut(&mut f))
        .collect();
    result
}

#[cfg(test)]
mod test_stats {
    use super::*;
    use ndarray::prelude::array;
    use ndarray_stats::QuantileExt;
    use noisy_float::types::{n64, N64};

    #[test]
    fn test_median_mut() {
        assert_eq!(
            median_mut(&mut Array::<usize, Ix1>::from(vec![])),
            Err(QuantileError::EmptyInput)
        );
        assert_eq!(median_mut(&mut array![1]), Ok(1));
        assert_eq!(median_mut(&mut array![1, 10]), Ok(5));
        assert_eq!(median_mut(&mut array![1, 10, 100]), Ok(10));
        assert_eq!(median_mut(&mut array![1, 10, 100, 1000]), Ok(55));

        assert_eq!(median_mut(&mut array![1.].mapv(n64)), Ok(n64(1.0)));
        assert_eq!(median_mut(&mut array![1., 10.].mapv(n64)), Ok(n64(5.5)));
        assert_eq!(median_mut(&mut array![1., 10., 100.].mapv(n64)), Ok(n64(10.0)));
        assert_eq!(median_mut(&mut array![1., 10., 100., 1000.].mapv(n64)), Ok(n64(55.0)));
    }

    #[test]
    fn test_median_array() {
        // Verify the library code and our work around for the stack overflow in it produce the same
        // results, note one method takes the matrix, the other the transpose of it.
        // this is to verify the change we made to merge_clusters when calculating the median is
        // still a-okay.
        let mut test_array = Array2::<N64>::from_shape_fn((20, 10), |(i, j)| (n64(i as f64) * n64(j as f64)));
        let mut test_array_transpose = Array2::<N64>::from_shape_fn((10, 20), |(i, j)| (n64(j as f64) * n64(i as f64)));
        let medians_old = test_array
            .quantile_axis_mut(Axis(0), n64(0.5), &Midpoint {})
            .expect("quantile failure!");
        let medians_new: Vec<N64> = median_array_rows_mut(&mut test_array_transpose).unwrap();
        assert_eq!(medians_old.len(), medians_new.len());
        medians_new
            .iter()
            .zip(medians_new.iter())
            .for_each(|(x, y)| assert_eq!(x, y));
    }
}
