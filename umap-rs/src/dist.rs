pub use crate::utils::Q;
use ndarray::Array1;
use simd_euclidean::Vectorized;

pub(crate) type DistanceFn = fn(x: &[Q], y: &[Q]) -> Q;
pub(crate) type DistanceGradFn = fn(x: &[Q], y: &[Q]) -> (Q, Array1<Q>);
pub(crate) type MetricToDistanceFn = fn(x: Q) -> Q;

#[derive(Clone, Copy)]
pub struct DistanceType(pub(crate) DistanceTypeImpl);

impl DistanceType {
    pub fn euclidean() -> Self {
        DistanceType(DistanceTypeImpl::Euclidean {
            dist: distance_functions::euclidean,
            metric: distance_functions::euclidean,
            metric2dist: |x| x,
        })
    }
    pub fn pearson() -> Self {
        // 1 - rho_{xy} is not a metric, but sqrt(1 - rho_{xy}) is,
        // see: https://arxiv.org/pdf/1908.06029.pdf
        DistanceType(DistanceTypeImpl::Euclidean {
            dist: distance_functions::pearson,
            metric: |x, y| distance_functions::pearson(x, y).sqrt(),
            metric2dist: |x| x.powi(2),
        })
    }
    pub fn cosine() -> Self {
        // similar to pearson, the sqrt is a metric
        DistanceType(DistanceTypeImpl::Other {
            dist: distance_functions::cosine,
            grad: distance_functions::euclidean_grad,
            metric: |x, y| distance_functions::cosine(x, y).sqrt(),
            metric2dist: |x| x.powi(2),
        })
    }
}

impl DistanceType {
    #[inline]
    pub(crate) fn metric(&self) -> DistanceFn {
        use DistanceTypeImpl::{Euclidean, Other};
        match self.0 {
            Euclidean { metric, .. } => metric,
            Other { metric, .. } => metric,
        }
    }
    #[inline]
    pub(crate) fn metric2dist(&self) -> MetricToDistanceFn {
        use DistanceTypeImpl::{Euclidean, Other};
        match self.0 {
            Euclidean { metric2dist, .. } => metric2dist,
            Other { metric2dist, .. } => metric2dist,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) enum DistanceTypeImpl {
    Euclidean {
        dist: DistanceFn,
        metric: DistanceFn,
        metric2dist: MetricToDistanceFn,
    },
    Other {
        dist: DistanceFn,
        grad: DistanceGradFn,
        metric: DistanceFn,
        metric2dist: MetricToDistanceFn,
    },
}

mod distance_functions {
    #![allow(dead_code)]
    use super::{Array1, Vectorized, Q};

    #[inline]
    pub fn cosine(x: &[Q], y: &[Q]) -> Q {
        let (s_xx, s_yy, s_xy) = x.iter().zip(y).fold((0., 0., 0.), |acc, (&x, &y)| {
            (acc.0 + x * x, acc.1 + y * y, acc.2 + x * y)
        });
        if s_xx == 0.0 && s_yy == 0.0 {
            0.0
        } else if s_xy == 0.0 {
            1.0
        } else {
            (1.0 - s_xy / (s_xx * s_yy).sqrt()).max(0.0)
        }
    }

    #[inline]
    pub fn cosine_grad(x: &[Q], y: &[Q]) -> (Q, Array1<Q>) {
        let (s_xx, s_yy, s_xy) = x.iter().zip(y).fold((0., 0., 0.), |acc, (&x, &y)| {
            (acc.0 + x * x, acc.1 + y * y, acc.2 + x * y)
        });
        if s_xx == 0.0 && s_yy == 0.0 {
            (0.0, Array1::zeros(x.len()))
        } else if s_xy == 0.0 {
            (1.0, Array1::zeros(x.len()))
        } else {
            let grad = x
                .iter()
                .zip(y)
                .map(|(&x, &y)| -(x * s_xy - y * s_xx) / (s_xx.powi(3) * s_yy).sqrt())
                .collect();
            let dist = (1.0 - s_xy / (s_xx * s_yy).sqrt()).max(0.0);
            (dist, grad)
        }
    }

    #[inline]
    pub fn euclidean(a: &[Q], b: &[Q]) -> Q {
        Vectorized::distance(a, b)
    }

    #[inline]
    pub fn euclidean_grad(x: &[Q], y: &[Q]) -> (Q, Array1<Q>) {
        let dist = Vectorized::distance(x, y);
        let grad = if dist == 0.0 {
            Array1::zeros(x.len())
        } else {
            x.iter()
                .zip(y)
                .map(|(&x, &y)| (x - y) / (1e-6 + dist))
                .collect::<Array1<Q>>()
        };
        (dist, grad)
    }

    #[inline]
    pub fn pearson(x: &[Q], y: &[Q]) -> Q {
        let (mu_x, mu_y) = x.iter().zip(y).fold((0., 0.), |acc, (&x, &y)| (acc.0 + x, acc.1 + y));
        let mu_x = mu_x / x.len() as Q;
        let mu_y = mu_y / y.len() as Q;

        let (s_xx, s_yy, s_xy) = x.iter().zip(y).fold((0., 0., 0.), |acc, (&x, &y)| {
            let s_x = x - mu_x;
            let s_y = y - mu_y;
            (acc.0 + s_x.powi(2), acc.1 + s_y.powi(2), acc.2 + s_x * s_y)
        });

        if s_xx == 0.0 && s_yy == 0.0 {
            0.0
        } else if s_xy == 0.0 {
            1.0
        } else {
            (1.0 - s_xy / (s_xx * s_yy).sqrt()).max(0.0)
        }
    }

    #[inline]
    pub fn pearson_grad(x: &[Q], y: &[Q]) -> (Q, Array1<Q>) {
        let (mu_x, mu_y) = x.iter().zip(y).fold((0., 0.), |acc, (&x, &y)| (acc.0 + x, acc.1 + y));
        let mu_x = mu_x / x.len() as Q;
        let mu_y = mu_y / y.len() as Q;

        let (s_xx, s_yy, s_xy) = x.iter().zip(y).fold((0., 0., 0.), |acc, (&x, &y)| {
            let s_x = x - mu_x;
            let s_y = y - mu_y;
            (acc.0 + s_x.powi(2), acc.1 + s_y.powi(2), acc.2 + s_x * s_y)
        });

        if s_xx == 0.0 && s_yy == 0.0 {
            (0.0, Array1::zeros(x.len()))
        } else if s_xy == 0.0 {
            (1.0, Array1::zeros(x.len()))
        } else {
            let dist = (1.0 - s_xy / (s_xx * s_yy).sqrt()).max(0.0);
            let grad: Array1<Q> = x
                .iter()
                .zip(y)
                .map(|(&x, &y)| ((x - mu_x) / s_xx - (y - mu_y) / s_xy) * dist)
                .collect();
            (dist, grad)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use ndarray_stats::CorrelationExt;

    #[test]
    fn test_pearson() {
        let data = [
            (18, 202),
            (32, 644),
            (25, 411),
            (60, 755),
            (12, 144),
            (25, 302),
            (50, 512),
            (15, 223),
            (22, 183),
            (30, 375),
        ];
        let (_x, _y): (Vec<_>, Vec<_>) = data.iter().map(|&(x, y)| (x as f64, y as f64)).unzip();

        let x = [1., 3., 5.];
        let y = [2., 4., 6.];
        let (dist, grad) = distance_functions::pearson_grad(&x, &y);
        println!("dist {dist:?}");
        println!("grad {grad:?}");
        let a = arr2(&[[1., 3., 5.], [2., 4., 6.]]);
        let corr = a.pearson_correlation().unwrap();
        println!("correlation {corr:?}");
        let epsilon = 1e-7;
        assert!(corr.abs_diff_eq(&arr2(&[[1., 1.], [1., 1.],]), epsilon));
    }

    #[test]
    fn test_cosine() {
        assert_eq!(distance_functions::cosine(&[3.0, 4.0], &[3.0, 4.0]), 0.0);
        assert_eq!(distance_functions::cosine(&[3.0, 4.0], &[-4.0, 3.0]), 1.0);
        assert_eq!(distance_functions::cosine(&[3.0, 4.0], &[-3.0, -4.0]), 2.0);
        assert_eq!(distance_functions::cosine(&[3.0, 4.0], &[4.0, -3.0]), 1.0);
    }
}
