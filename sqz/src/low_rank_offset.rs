use crate::{AdaptiveMat, AdaptiveVec, MatrixMap};
use ndarray::linalg::Dot;
use ndarray::{Array2, ArrayBase, Data, Ix2};
// use num_traits::{FromPrimitive, NumAssign};
use std::ops::Deref;

/// An efficient representation of a matrix
/// `A = mat + u * v` where `mat` is sparse,
/// `u` is a column vector and `v` is row vector.
/// `u * v` forms a rank-1 'offset' to the sparse
/// matrix, which can efficiently tracked without
/// de-sparsifying `mat`.
pub struct LowRankOffset<D, M> {
    mat: AdaptiveMat<f64, D, M>,
    u: Array2<f64>,
    v: Array2<f64>,
}

impl<D, M> LowRankOffset<D, M>
where
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, f64>,
{
    /// Create a `LowRankOffset` matrix that represents the sum of the
    /// sparse matrix `mat` and the low rank matrix product of `u * v`.
    /// The shape of the offset matrix `u*v` must match `mat`.
    pub fn new(mat: AdaptiveMat<f64, D, M>, u: Array2<f64>, v: Array2<f64>) -> LowRankOffset<D, M> {
        assert_eq!(mat.rows(), u.shape()[0]);
        assert_eq!(mat.cols(), v.shape()[1]);
        assert_eq!(u.shape()[1], v.shape()[0]);

        LowRankOffset { mat, u, v }
    }

    /// number of rows of the LowRankOffset
    pub fn rows(&self) -> usize {
        self.mat.rows()
    }

    /// number of columns of the LowRankOffset
    pub fn cols(&self) -> usize {
        self.mat.cols()
    }

    /// shape of the LowRankOffset
    pub fn shape(&self) -> [usize; 2] {
        [self.rows(), self.cols()]
    }

    /// A reference to the inner sparse matrix.
    pub fn inner_sparse(&self) -> &AdaptiveMat<f64, D, M> {
        &self.mat
    }

    /// Convert the matrix to a dense matrix
    pub fn to_dense(&self) -> Array2<f64> {
        self.u.dot(&self.v) + self.mat.dot(&Array2::<f64>::eye(self.mat.cols()))
    }

    /// transpose of the LowRankOffset
    pub fn t(&self) -> LowRankOffset<&[AdaptiveVec], M::T> {
        let mat = self.mat.t();
        let u = self.v.t().to_owned();
        let v = self.u.t().to_owned();
        LowRankOffset { mat, u, v }
    }
}

impl<'a, D, M, DS> Dot<ArrayBase<DS, Ix2>> for LowRankOffset<D, M>
where
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, f64>,
    DS: 'a + Data<Elem = f64>,
{
    type Output = Array2<f64>;

    fn dot(&self, rhs: &ArrayBase<DS, Ix2>) -> Array2<f64> {
        let mut res = self.mat.dot(rhs);
        res += &self.u.dot(&self.v.dot(rhs));
        res
    }
}

impl<'a, D, M, DS> Dot<LowRankOffset<D, M>> for ArrayBase<DS, Ix2>
where
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, f64>,
    DS: 'a + Data<Elem = f64>,
{
    type Output = Array2<f64>;

    fn dot(&self, rhs: &LowRankOffset<D, M>) -> Array2<f64> {
        let mut res = self.dot(&rhs.mat);
        res += &(self.dot(&rhs.u).dot(&rhs.v));
        res
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::mat::test as mat_test;
    use ndarray::{ArrayView, Dimension};
    use ndarray_rand::RandomExt;
    use rand::distributions::uniform::Uniform;
    use rand::prelude::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct F64Map;

    impl MatrixMap<u32, f64> for F64Map {
        type T = Self;

        #[inline(always)]
        fn map(&self, v: u32, _: usize, _: usize) -> f64 {
            v as f64
        }

        fn t(&self) -> Self::T {
            self.clone()
        }
    }

    // stolen from ndarray - not currently exported.
    fn assert_close<D>(a: ArrayView<f64, D>, b: ArrayView<f64, D>)
    where
        D: Dimension,
    {
        let diff = (&a - &b).mapv_into(f64::abs);

        let rtol = 1e-7;
        let atol = 1e-12;
        let crtol = b.mapv(|x| x.abs() * rtol);
        let tol = crtol + atol;
        let tol_m_diff = &diff - &tol;
        let maxdiff = tol_m_diff.fold(f64::NAN, |x, y| f64::max(x, *y));
        println!("diff offset from tolerance level= {maxdiff:.2e}");
        if maxdiff > 0. {
            println!("{a:.4?}");
            println!("{b:.4?}");
            panic!("results differ");
        }
    }

    #[test]
    fn test_low_rank_dot() {
        let mut rng = Pcg64Mcg::seed_from_u64(42);

        for (rows, cols, mat) in mat_test::random_matrices(1000, 10) {
            // Convert mat to f64 output
            let mat = mat.set_map(F64Map);

            let rank = rng.gen_range(1..5);

            let unif = Uniform::new(-1.0, 1.0);
            let u = Array2::random_using((rows, rank), unif, &mut rng);
            let v = Array2::random_using((rank, cols), unif, &mut rng);

            let lr_sparse = LowRankOffset { mat, u, v };
            let lr_dense = lr_sparse.to_dense();

            let query = Array2::random_using((cols, 16), unif, &mut rng);

            let q1 = lr_sparse.dot(&query);
            let q2 = lr_dense.dot(&query);
            assert_close(q1.view(), q2.view());

            let lr_sparse = lr_sparse.t();
            let lr_dense = lr_dense.t();

            let q1 = query.t().dot(&lr_sparse);
            let q2 = query.t().dot(&lr_dense);
            assert_close(q1.view(), q2.view());
        }
    }
}
