use ndarray::prelude::*;
use num_traits::Num;
use std::marker::PhantomData;

/// Trait to apply a mapping function to each non-zero cell of the matrix, which
/// can be a function of the row and column of the cell.
pub trait MatrixMap<In, Out>: Clone + PartialEq + Eq {
    /// return type of the transpose operator
    type T: MatrixMap<In, Out>;

    /// How to map a value
    fn map(&self, v: In, r: usize, c: usize) -> Out;

    /// How to transpose the map
    // TODO: Try replacing self::T with TransposeMap<...>. Need to check that transposing
    // again reverts the first transpose.
    // If this works, could also add fn compose_with(&self, other_map)
    fn t(&self) -> Self::T;
}

// /// Empty MatrixMap: identity transformation, possibly with change of type
// impl<N> MatrixMap<N, N> for () {
//     type T = ();

//     #[inline(always)]
//     fn map(&self, v: N, _: usize, _: usize) -> N {
//         v
//     }

//     #[inline(always)]
//     fn t(&self) -> Self::T {
//         ()
//     }
// }

#[derive(Clone)]
/// Transpose of a MatrixMap
/// Note, can't derive PartialEq because of In and Out; for details see
/// <https://github.com/rust-lang/rust/issues/52079>
/// <https://github.com/rust-lang/rust/issues/26925>
pub struct TransposeMap<In, Out, M> {
    base: M,
    _in: PhantomData<In>,
    _out: PhantomData<Out>,
}

impl<In, Out, M> TransposeMap<In, Out, M>
where
    In: Clone,
    Out: Clone,
    M: MatrixMap<In, Out>,
{
    /// Create a new TransposeMap
    pub fn new(base: M) -> Self {
        TransposeMap {
            base,
            _in: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<In, Out, M> MatrixMap<In, Out> for TransposeMap<In, Out, M>
where
    In: Clone,
    Out: Clone,
    M: MatrixMap<In, Out>,
{
    type T = M;

    #[inline(always)]
    fn map(&self, v: In, r: usize, c: usize) -> Out {
        self.base.map(v, c, r)
    }

    fn t(&self) -> Self::T {
        self.base.clone()
    }
}

impl<In, Out, M> PartialEq for TransposeMap<In, Out, M>
where
    M: MatrixMap<In, Out>,
{
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
    }
}

impl<In, Out, M> Eq for TransposeMap<In, Out, M> where M: MatrixMap<In, Out> {}

/// MatrixMap that only changes type. Useful for composing with other maps
/// Note: can't derive PartialEq; for details see
/// <https://github.com/rust-lang/rust/issues/52079>
/// <https://github.com/rust-lang/rust/issues/26925>
#[derive(Clone, Debug)]
pub struct MatrixIntoMap<In, Out> {
    _in: PhantomData<In>,
    _out: PhantomData<Out>,
}

impl<In: Clone + Into<Out>, Out: Clone> MatrixIntoMap<In, Out> {
    /// Create a new MatrixIntoMap
    pub fn new() -> Self {
        Self {
            _in: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<In: Clone + Into<Out>, Out: Clone> Default for MatrixIntoMap<In, Out> {
    fn default() -> Self {
        Self::new()
    }
}

impl<In: Clone + Into<Out>, Out: Clone> MatrixMap<In, Out> for MatrixIntoMap<In, Out> {
    type T = TransposeMap<In, Out, Self>;

    #[inline(always)]
    fn map(&self, v: In, _r: usize, _c: usize) -> Out {
        v.into()
    }

    fn t(&self) -> Self::T {
        TransposeMap::new(self.clone())
    }
}

impl<In, Out> PartialEq for MatrixIntoMap<In, Out> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<In, Out> Eq for MatrixIntoMap<In, Out> {}

/// Composition of two MatrixMap objects
/// Note: can't derive PartialEq; for details see
/// <https://github.com/rust-lang/rust/issues/52079>
/// <https://github.com/rust-lang/rust/issues/26925>
#[derive(Clone)]
pub struct ComposedMap<N1, N2, N3, N4, Inner, Outer> {
    inner: Inner,
    outer: Outer,
    _n1: PhantomData<N1>,
    _n2: PhantomData<N2>,
    _n3: PhantomData<N3>,
    _n4: PhantomData<N4>,
}

// Note: trait bounds here are repeated here and in the `impl MatixMap` block for ease of use.
// Otherwise, doing just ComposedMap::new(a, b) will
impl<N1, N2, N3, N4, Inner, Outer> ComposedMap<N1, N2, N3, N4, Inner, Outer>
where
    N1: Clone,
    N2: Clone + Into<N3>,
    N3: Clone,
    N4: Clone,
    Inner: MatrixMap<N1, N2>,
    Outer: MatrixMap<N3, N4>,
{
    /// Create a new ComposedMap
    pub fn new(inner: Inner, outer: Outer) -> Self {
        ComposedMap {
            inner,
            outer,
            _n1: PhantomData,
            _n2: PhantomData,
            _n3: PhantomData,
            _n4: PhantomData,
        }
    }
}

impl<N1, N2, N3, N4, Inner, Outer> MatrixMap<N1, N4> for ComposedMap<N1, N2, N3, N4, Inner, Outer>
where
    N1: Clone,
    N2: Clone + Into<N3>,
    N3: Clone,
    N4: Clone,
    Inner: MatrixMap<N1, N2>,
    Outer: MatrixMap<N3, N4>,
{
    type T = TransposeMap<N1, N4, Self>;

    #[inline(always)]
    fn map(&self, v: N1, r: usize, c: usize) -> N4 {
        self.outer.map(self.inner.map(v, r, c).into(), r, c)
    }

    fn t(&self) -> Self::T {
        TransposeMap::new(self.clone())
    }
}

impl<N1, N2, N3, N4, Inner, Outer> PartialEq for ComposedMap<N1, N2, N3, N4, Inner, Outer>
where
    Inner: MatrixMap<N1, N2>,
    Outer: MatrixMap<N3, N4>,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner && self.outer == other.outer
    }
}

impl<N1, N2, N3, N4, Inner, Outer> Eq for ComposedMap<N1, N2, N3, N4, Inner, Outer>
where
    Inner: MatrixMap<N1, N2>,
    Outer: MatrixMap<N3, N4>,
{
}

/// MatrixMap that multiplies columns or rows by scale factors
/// Note: can't derive PartialEq; for details see
/// <https://github.com/rust-lang/rust/issues/52079>
/// <https://github.com/rust-lang/rust/issues/26925>
#[derive(Clone)]
pub struct ScaleAxis {
    axis: Axis,
    scale_factors: Array1<f64>,
}

impl ScaleAxis {
    /// Create a new ScaleAxis map, with scale factors provided as f64
    pub fn new(axis: Axis, scale_factors: Array1<f64>) -> Self {
        assert!(axis.index() < 2, "Only implemented for 2D arrays.");
        // if let Axis(i) = axis {
        //     assert!(i < 2, "Only implemented for 2D arrays.");
        // }
        Self { axis, scale_factors }
    }

    /// Create a new ScaleAxis map, with scale factors provided as f64
    pub fn from_f64(axis: Axis, scale_factors: Array1<f64>) -> Self {
        Self::new(axis, scale_factors)
    }
}

impl MatrixMap<f64, f64> for ScaleAxis {
    type T = TransposeMap<f64, f64, Self>;

    #[inline(always)]
    fn map(&self, v: f64, r: usize, c: usize) -> f64 {
        match self.axis {
            Axis(0) => self.scale_factors[r] * v,
            Axis(1) => self.scale_factors[c] * v,
            _ => unreachable!(),
        }
    }

    fn t(&self) -> Self::T {
        TransposeMap::new(self.clone())
    }
}

impl PartialEq for ScaleAxis {
    fn eq(&self, other: &Self) -> bool {
        self.axis == other.axis && self.scale_factors == other.scale_factors
    }
}

impl Eq for ScaleAxis {}

/// MatrixMap that applies a scalar function f to each entry, where f(0) = 0.
#[derive(Clone)]
pub struct ScalarMap<In, Out, F> {
    f: Box<F>,
    _in: PhantomData<In>,
    _out: PhantomData<Out>,
}

impl<In, Out, F> ScalarMap<In, Out, F>
where
    In: Copy + Num,
    Out: Copy + Num,
    F: Clone + Fn(In) -> Out,
{
    /// Create a new ScalarMap
    pub fn new(f: F) -> Self {
        assert!(f(In::zero()) == Out::zero(), "Function must map 0 to 0");
        Self {
            f: Box::new(f),
            _in: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<In, Out, F> MatrixMap<In, Out> for ScalarMap<In, Out, F>
where
    In: Copy + Num,
    Out: Copy + Num,
    F: Clone + Fn(In) -> Out,
{
    type T = TransposeMap<In, Out, Self>;

    #[inline(always)]
    fn map(&self, v: In, _r: usize, _c: usize) -> Out {
        (self.f)(v)
    }

    fn t(&self) -> Self::T {
        TransposeMap::new(self.clone())
    }
}

impl<In, Out, F> PartialEq for ScalarMap<In, Out, F>
where
    In: Copy + Num,
    Out: Copy + Num,
    F: Clone + Fn(In) -> Out,
{
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<In, Out, F> Eq for ScalarMap<In, Out, F>
where
    In: Copy + Num,
    Out: Copy + Num,
    F: Clone + Fn(In) -> Out,
{
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mat::test as mat_test;
    use ndarray::{ArrayView, Dimension};
    use rand::distr::Uniform;
    use rand::prelude::{SeedableRng, *};
    use rand::rngs::SmallRng;

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

    fn check<In, Out, M, A>(orig: &Array2<In>, matrix_map: &M, transformed: &Array2<A>)
    where
        In: Copy,
        Out: Copy + Into<f64>,
        M: MatrixMap<In, Out>,
        A: Copy + Into<f64>,
    {
        check_matrix_map(orig, matrix_map, transformed);
        check_transpose(&orig.t().to_owned(), &matrix_map.t(), &transformed.t().to_owned());
    }

    /// check that `orig` mapped through `matrix_map` is close to `transformed`
    fn check_matrix_map<In, Out, M, A>(orig: &Array2<In>, matrix_map: &M, transformed: &Array2<A>)
    where
        In: Copy,
        Out: Copy + Into<f64>,
        M: MatrixMap<In, Out>,
        A: Copy + Into<f64>,
    {
        let matrix2: Array2<f64> = transformed.map(|&x| x.into());
        let mut matrix1_mapped = Array2::zeros(orig.dim());
        for ((r, c), x) in matrix1_mapped.indexed_iter_mut() {
            *x = matrix_map.map(orig[(r, c)], r, c).into();
        }
        assert_close(matrix1_mapped.view(), matrix2.view());
    }

    /// check that `orig.t()` mapped through `matrix_map.t()` is close to `transformed.t()`
    fn check_transpose<In, Out, M, A>(orig: &Array2<In>, matrix_map: &M, transformed: &Array2<A>)
    where
        In: Copy,
        Out: Copy + Into<f64>,
        M: MatrixMap<In, Out>,
        A: Copy + Into<f64>,
    {
        let orig = orig.t().to_owned();
        let transformed = transformed.t().to_owned();
        let matrix_map = matrix_map.t();
        check_matrix_map(&orig, &matrix_map, &transformed);
    }

    #[test]
    fn test_into() {
        let orig: Array2<u32> = array![[1, 2, 3], [4, 5, 6]];
        let transformed: Array2<f64> = orig.map(|&x| x.into());
        let matrix_map = MatrixIntoMap::<u32, f64>::new();

        check(&orig, &matrix_map, &transformed);
    }

    #[test]
    fn test_scalar_map() {
        let orig: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let transformed: Array2<f64> = orig.map(|&x| (x + 1.0).ln());
        let matrix_map = ScalarMap::new(|x: f64| (x + 1.0).ln());

        check(&orig, &matrix_map, &transformed);
    }

    #[test]
    fn test_scale_axis() {
        let orig: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let scale_factors = array![0.0, 1.0, 2.0];

        let scale_rows = ScaleAxis::new(Axis(0), scale_factors.clone());
        let expected = array![[0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [14.0, 16.0, 18.0]];
        check(&orig, &scale_rows, &expected);

        let scale_cols = ScaleAxis::new(Axis(1), scale_factors);
        let expected = array![[0.0, 2.0, 6.0], [0.0, 5.0, 12.0], [0.0, 8.0, 18.0]];
        check(&orig, &scale_cols, &expected);
    }

    #[test]
    fn test_composed_map() {
        let orig: Array2<f64> = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];

        let scale_rows = ScaleAxis::new(Axis(0), array![1.0, 2.0, 3.0]);
        let square = ScalarMap::new(|x: f64| x.powi(2));

        let scale_then_square = ComposedMap::new(scale_rows.clone(), square.clone());
        let expected = array![[1.0, 4.0, 9.0], [16.0, 36.0, 64.0], [81.0, 144.0, 225.0]];
        check(&orig, &scale_then_square, &expected);

        let square_then_scale = ComposedMap::new(square, scale_rows);
        let expected = array![[1.0, 4.0, 9.0], [8.0, 18.0, 32.0], [27.0, 48.0, 75.0]];
        check(&orig, &square_then_scale, &expected);
    }

    #[test]
    fn test_combined_matrix_maps() {
        // generate test matrices
        let mats = mat_test::random_matrices(1000, 100);
        let rng = &mut SmallRng::seed_from_u64(0);

        for (_rows, _cols, orig) in mats {
            println!("{orig:?}");
            let mut orig = orig.to_dense();
            let tranformed = orig.clone();
            let matrix_map = MatrixIntoMap::<u32, u32>::new();

            // identity map
            check(&orig, &matrix_map, &tranformed);
            check_transpose(&orig, &matrix_map, &tranformed);

            // transposing the identity map doesn't do anything
            let matrix_map = matrix_map.t();
            check(&orig, &matrix_map, &tranformed);

            // compose with scalar map
            let log1p = |x: u32| (x as f64 + 1_f64).ln();

            let log1p_map: ScalarMap<_, _, _> = ScalarMap::new(log1p);
            let matrix_map = ComposedMap::new(matrix_map, log1p_map);
            let mut tranformed = tranformed.mapv(log1p);

            check(&orig, &matrix_map, &tranformed);

            // transpose
            orig = orig.t().to_owned();
            let matrix_map = matrix_map.t();
            tranformed = tranformed.t().to_owned();

            check(&orig, &matrix_map, &tranformed);

            // compose with row scaling
            println!("compose with row scaling");
            let unif_iter = rng.sample_iter(Uniform::new(0_f64, 1_f64).unwrap());
            let row_scaling = Array1::from_iter(unif_iter.take(orig.shape()[0]));

            let scale_rows = ScaleAxis::new(Axis(0), row_scaling.clone());
            let matrix_map = ComposedMap::new(matrix_map, scale_rows);
            for (i, mut row_mut) in tranformed.axis_iter_mut(Axis(0)).enumerate() {
                row_mut *= row_scaling[i];
            }

            check(&orig, &matrix_map, &tranformed);

            // transpose
            orig = orig.t().to_owned();
            let matrix_map = matrix_map.t();
            tranformed = tranformed.t().to_owned();

            check(&orig, &matrix_map, &tranformed);

            // compose with column scaling
            println!("compose with column scaling");
            let unif_iter = rng.sample_iter(Uniform::new(0_f64, 1_f64).unwrap());
            let col_scaling = Array1::from_iter(unif_iter.take(orig.shape()[1]));

            let scale_columns = ScaleAxis::new(Axis(1), col_scaling.clone());
            let matrix_map = ComposedMap::new(matrix_map, scale_columns);
            for (i, mut col_mut) in tranformed.axis_iter_mut(Axis(1)).enumerate() {
                col_mut *= col_scaling[i];
            }

            check(&orig, &matrix_map, &tranformed);
        }
    }
}
