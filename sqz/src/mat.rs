use crate::low_rank_offset::LowRankOffset;
use crate::matrix_map::{ComposedMap, MatrixIntoMap, MatrixMap, ScalarMap, ScaleAxis};
use crate::prod;
use crate::vec::{AbstractVec, AdaptiveVec};
use anyhow::Error;
use itertools::Itertools;
use log::info;
use ndarray::{linalg::Dot, Array, Array1, Array2, ArrayBase, Axis, Ix1, Ix2};
use num_traits::{FromPrimitive, Num, NumAssignRef, Zero};
use rayon::prelude::*;
use snoop::{CancelProgress, NoOpSnoop};
use std::ops::{AddAssign, Deref, Mul};
use std::{collections::HashSet, marker::PhantomData, time::Instant};

/// Trait for numeric types suitable to be used as an output of AdaptiveMat.
pub trait AdaptiveMatNum: Copy + PartialOrd + FromPrimitive + NumAssignRef + Into<f64> {}

impl AdaptiveMatNum for u32 {}
impl AdaptiveMatNum for f32 {}
impl AdaptiveMatNum for f64 {}

/// Matrix formed from a collection of `AdaptiveVec` values. The matrix map, M, is an
/// implementation of the `MatrixMap` trait that permits mapping each non-zero value to a
/// new value.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct AdaptiveMat<N = u32, D = Vec<AdaptiveVec>, M = MatrixIntoMap<u32, N>> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) storage: sprs::CompressedStorage,
    pub(crate) data: D,
    pub(crate) matrix_map: M,
    _n: PhantomData<N>,
}

/// View of an `AdaptiveMat`
pub type AdaptiveMatView<'a, N = u32, M = MatrixIntoMap<u32, N>> = AdaptiveMat<N, &'a [AdaptiveVec], M>;

/// `AdaptiveMat` that owns its data
pub type AdaptiveMatOwned<N = u32, M = MatrixIntoMap<u32, N>> = AdaptiveMat<N, Vec<AdaptiveVec>, M>;

impl<N, D> AdaptiveMat<N, D>
where
    N: AdaptiveMatNum,
    u32: Into<N>,
    D: Deref<Target = [AdaptiveVec]>,
{
    /// Create a new `AdaptiveMat` with output type `N` and the default `MatrixMap` that
    /// calls into() on each element. Must have `u32: Into<N>`
    pub fn new(rows: usize, cols: usize, storage: sprs::CompressedStorage, data: D) -> AdaptiveMat<N, D> {
        AdaptiveMat::new_with_map(rows, cols, storage, data, MatrixIntoMap::new())
    }
}

impl<N> AdaptiveMatOwned<N>
where
    N: AdaptiveMatNum,
    u32: Into<N>,
{
    /// Create an `AdaptiveMat` from a `sprs::CsMat`
    pub fn from_csmat<T: Clone + Into<u32>, I: sprs::SpIndex + num_traits::ToPrimitive>(
        mat: &sprs::CsMatI<T, I>,
    ) -> AdaptiveMatOwned<N> {
        let (rows, cols) = mat.shape();

        let vec_len = if mat.storage() == sprs::CSC { rows } else { cols };

        let mut data = Vec::new();

        let mut tmp_indices: Vec<u32> = Vec::new();
        let mut tmp_values: Vec<u32> = Vec::new();

        for v in mat.outer_iterator() {
            tmp_indices.clear();
            tmp_indices.extend(v.indices().iter().cloned().map(|v: I| v.to_u32().unwrap()));

            tmp_values.clear();
            tmp_values.extend(v.data().iter().cloned().map(std::convert::Into::into));

            let compressed_vec = AdaptiveVec::new(vec_len, &tmp_values, &tmp_indices);
            data.push(compressed_vec);
        }

        AdaptiveMat::new(rows, cols, mat.storage(), data)
    }
}

type AdaptiveMatPartition<N, M> = (AdaptiveMatOwned<N, M>, AdaptiveMatOwned<N, M>, Vec<usize>, Vec<usize>);

impl<N, D, M> AdaptiveMat<N, D, M>
where
    N: AdaptiveMatNum,
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, N>,
{
    /// Create a new `AdaptiveMat` with a specific `MatrixMap`
    pub fn new_with_map(
        rows: usize,
        cols: usize,
        storage: sprs::CompressedStorage,
        data: D,
        matrix_map: M,
    ) -> AdaptiveMat<N, D, M> {
        let mut counts = [0; 5];
        for vec in data.iter() {
            match vec {
                AdaptiveVec::D4(_) => counts[0] += 1,
                AdaptiveVec::D8(_) => counts[1] += 1,
                AdaptiveVec::D16(_) => counts[2] += 1,
                AdaptiveVec::S4(_) => counts[3] += 1,
                AdaptiveVec::S8(_) => counts[4] += 1,
            }
        }
        let _n = PhantomData;
        AdaptiveMat {
            rows,
            cols,
            storage,
            data,
            matrix_map,
            _n,
        }
    }

    /// Number of non-zero elements in the matrix
    pub fn nnz(&self) -> usize {
        self.data.par_iter().map(AdaptiveVec::nnz).sum()
    }

    /// Number of columns in the matrix
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of rows in the matrix
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Shape in [rows, cols] of the matrix
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    /// Is the matrix laid out row-major / compressed sparse row (CSR) layout.
    pub fn is_csr(&self) -> bool {
        self.storage == sprs::CSR
    }

    /// Is the matrix laid out column-major / compressed sparse column (CSC) layout.
    pub fn is_csc(&self) -> bool {
        self.storage == sprs::CSC
    }

    /// Iterate over the outer dimension of the matrix
    pub fn iter(&self) -> impl Iterator<Item = &AdaptiveVec> {
        self.data.iter()
    }

    /// Convert the matrix to a dense 2D array
    pub fn to_dense(&self) -> Array2<N> {
        let mut arr = Array2::zeros((self.rows(), self.cols()));

        if self.storage == sprs::CSR {
            for (row, vec) in self.data.iter().enumerate() {
                vec.foreach(|col, value| arr[(row, col)] = self.matrix_map.map(value, row, col));
            }
        } else {
            for (col, vec) in self.data.iter().enumerate() {
                vec.foreach(|row, value| arr[(row, col)] = self.matrix_map.map(value, row, col));
            }
        }

        arr
    }

    /// Create a `sprs::CsMat<N>` from this matrix
    pub fn to_csmat(&self) -> sprs::CsMat<N> {
        let nnz = self.data.iter().map(|v| ada_expand!(v, e, e.nnz())).sum();

        let mut indptr = Vec::with_capacity(self.data.len() + 1);
        let mut indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        let mut total_len: usize = 0;
        indptr.push(total_len);

        for (outer, vec) in self.data.iter().enumerate() {
            vec.foreach(|inner, v| {
                indices.push(inner);
                let (r, c) = if self.storage == sprs::CSR {
                    (outer, inner)
                } else {
                    (inner, outer)
                };

                let new_value = self.matrix_map.map(v, r, c);
                values.push(new_value);
            });

            total_len += ada_expand!(vec, e, e.nnz());
            indptr.push(total_len);
        }

        if self.storage == sprs::CSR {
            sprs::CsMat::new((self.rows(), self.cols()), indptr, indices, values)
        } else {
            sprs::CsMat::new_csc((self.rows(), self.cols()), indptr, indices, values)
        }
    }

    /// Make a view of the matrix
    pub fn view(&self) -> AdaptiveMatView<N, M> {
        AdaptiveMat::new_with_map(self.rows, self.cols, self.storage, &self.data, self.matrix_map.clone())
    }

    /// View of the underlying sparse matrix
    pub fn view_base_mat(&self) -> AdaptiveMatView<u32> {
        self.view().set_map(MatrixIntoMap::new())
    }

    /// Convert the underlying sparse matrix to a 2D array of u32
    pub fn base_mat_dense(&self) -> Array2<u32> {
        self.view_base_mat().to_dense()
    }

    /// Convert the underlying sparse matrix to a `sparse::CsMat<u32>`
    pub fn base_mat_csc(&self) -> sprs::CsMat<u32> {
        self.view_base_mat().to_csmat()
    }

    /// View of the transposed matrix
    pub fn t(&self) -> AdaptiveMatView<N, M::T> {
        AdaptiveMat::new_with_map(
            self.cols,
            self.rows,
            self.storage.other_storage(),
            &self.data,
            self.matrix_map.t(),
        )
    }

    /// Return the mean along an axis
    pub fn mean_axis(&self, axis: Axis) -> Array1<f64> {
        let m = self.shape()[axis.index()] as f64;
        self.sum_axis::<f64>(axis).mapv_into(|v| v / m)
    }

    /// Return the means of each row, using only the column indices provided
    pub fn mean_rows(&self, cols: &[usize]) -> Array1<f64> {
        let m = cols.len() as f64;
        self.sum_rows::<f64>(cols).mapv_into(|v| v / m)
    }

    /// Return the mean and variance along an axis
    pub fn mean_var_axis(&self, axis: Axis) -> (Array1<f64>, Array1<f64>) {
        let sz = if axis.index() == 0 { self.cols() } else { self.rows() };
        let mut means = Array1::<f64>::zeros((sz,));
        let mut vars = Array1::<f64>::zeros((sz,));

        if self.storage == sprs::CSR {
            if axis.index() == 0 {
                for (row, vec) in self.data.iter().enumerate() {
                    vec.foreach(|col, value| {
                        means[col] += self.matrix_map.map(value, row, col).into();
                        vars[col] += self.matrix_map.map(value, row, col).into().powi(2);
                    });
                }
            } else {
                for (row, vec) in self.data.iter().enumerate() {
                    vec.foreach(|col, value| {
                        means[row] += self.matrix_map.map(value, row, col).into();
                        vars[row] += self.matrix_map.map(value, row, col).into().powi(2);
                    });
                }
            }
        } else if axis.index() == 0 {
            for (col, vec) in self.data.iter().enumerate() {
                vec.foreach(|row, value| {
                    means[col] += self.matrix_map.map(value, row, col).into();
                    vars[col] += self.matrix_map.map(value, row, col).into().powi(2);
                });
            }
        } else {
            for (col, vec) in self.data.iter().enumerate() {
                vec.foreach(|row, value| {
                    means[row] += self.matrix_map.map(value, row, col).into();
                    vars[row] += self.matrix_map.map(value, row, col).into().powi(2);
                });
            }
        }

        // V[X] = E[X^2] - E[X]^2
        let m = self.shape()[axis.index()] as f64;
        for (mean, var) in means.iter_mut().zip(vars.iter_mut()) {
            *mean /= m;
            *var = *var / m - mean.powi(2)
        }

        (means, vars)
    }

    /// Return the means and vars of each row, using only the provided sorted column indices
    pub fn mean_var_rows(&self, cols: &[usize]) -> (Array1<f64>, Array1<f64>) {
        let sz = self.rows();
        let mut means = Array1::<f64>::zeros((sz,));
        let mut vars = Array1::<f64>::zeros((sz,));

        if self.storage == sprs::CSR {
            for (row, vec) in self.data.iter().enumerate() {
                // setup coordinating iterators
                let mut iter = cols.iter();
                let mut next = iter.next();
                vec.foreach(|col, value| loop {
                    match next {
                        Some(&c) if c < col => {
                            next = iter.next();
                        }
                        Some(&c) if c == col => {
                            means[row] += self.matrix_map.map(value, row, col).into();
                            vars[row] += self.matrix_map.map(value, row, col).into().powi(2);
                            next = iter.next();
                        }
                        _ => return,
                    }
                })
            }
        } else {
            for &col in cols {
                self.data[col].foreach(|row, value| {
                    means[row] += self.matrix_map.map(value, row, col).into();
                    vars[row] += self.matrix_map.map(value, row, col).into().powi(2);
                })
            }
        }

        // V[X] = E[X^2] - E[X]^2
        let m = cols.len() as f64;
        for (mean, var) in means.iter_mut().zip(vars.iter_mut()) {
            *mean /= m;
            *var = *var / m - mean.powi(2)
        }

        (means, vars)
    }

    /// Return the sum along an axis
    pub fn sum_axis<O>(&self, axis: Axis) -> Array1<O>
    where
        N: Into<O>,
        O: AddAssign + Clone + Zero,
    {
        let sz = if axis.index() == 0 { self.cols() } else { self.rows() };
        let mut arr = Array1::<O>::zeros((sz,));

        if self.storage == sprs::CSR {
            if axis.index() == 0 {
                for (row, vec) in self.data.iter().enumerate() {
                    vec.foreach(|col, value| arr[col] += self.matrix_map.map(value, row, col).into());
                }
            } else {
                for (row, vec) in self.data.iter().enumerate() {
                    vec.foreach(|col, value| arr[row] += self.matrix_map.map(value, row, col).into());
                }
            }
        } else if axis.index() == 0 {
            for (col, vec) in self.data.iter().enumerate() {
                vec.foreach(|row, value| arr[col] += self.matrix_map.map(value, row, col).into());
            }
        } else {
            for (col, vec) in self.data.iter().enumerate() {
                vec.foreach(|row, value| arr[row] += self.matrix_map.map(value, row, col).into());
            }
        }

        arr
    }

    /// Return the variance along an axis
    pub fn var_axis(&self, axis: Axis) -> Array1<f64> {
        self.mean_var_axis(axis).1
    }

    /// Return the sum of cols, which must be sorted
    pub fn sum_cols<O>(&self, cols: &[usize]) -> Array1<O>
    where
        N: Into<O>,
        O: AddAssign + Clone + Zero,
    {
        let mut arr = Array1::<O>::zeros((cols.len(),));

        if self.storage == sprs::CSR {
            for (row, vec) in self.data.iter().enumerate() {
                // setup coordinating iterators
                let mut iter = cols.iter().enumerate();
                let mut next = iter.next();
                vec.foreach(|col, value| loop {
                    match next {
                        Some((_, &c)) if c < col => {
                            next = iter.next();
                        }
                        Some((i, &c)) if c == col => {
                            arr[i] += self.matrix_map.map(value, row, col).into();
                            next = iter.next();
                        }
                        _ => return,
                    }
                })
            }
        } else {
            for (i, &col) in cols.iter().enumerate() {
                self.data[col].foreach(|row, value| arr[i] += self.matrix_map.map(value, row, col).into());
            }
        }

        arr
    }

    /// Return the sum of rows over cols, which must be sorted
    pub fn sum_rows<O>(&self, cols: &[usize]) -> Array1<O>
    where
        N: Into<O>,
        O: AddAssign + Clone + Zero,
    {
        let mut arr = Array1::<O>::zeros((self.rows(),));

        if self.storage == sprs::CSR {
            for (row, vec) in self.data.iter().enumerate() {
                // setup coordinating iterators
                let mut iter = cols.iter();
                let mut next = iter.next();
                vec.foreach(|col, value| loop {
                    match next {
                        Some(&c) if c < col => {
                            next = iter.next();
                        }
                        Some(&c) if c == col => {
                            arr[row] += self.matrix_map.map(value, row, col).into();
                            next = iter.next();
                        }
                        _ => return,
                    }
                })
            }
        } else {
            for &col in cols.iter() {
                self.data[col].foreach(|row, value| arr[row] += self.matrix_map.map(value, row, col).into());
            }
        }

        arr
    }

    /// Sum across two sets of column indices simultaneously
    pub fn sum_rows_dual<O>(&self, cols1: &[usize], cols2: &[usize]) -> (Array1<O>, Array1<O>)
    where
        N: Into<O>,
        O: AddAssign + Clone + Copy + Zero,
    {
        let snoop = NoOpSnoop {};
        self.sum_rows_dual_with_cancellation(snoop, cols1, cols2).unwrap()
    }

    /// Sum across two sets of column indices simultaneously with cancellation
    pub fn sum_rows_dual_with_cancellation<O, C>(
        &self,
        mut snoop: C,
        cols1: &[usize],
        cols2: &[usize],
    ) -> Result<(Array1<O>, Array1<O>), Error>
    where
        N: Into<O>,
        O: AddAssign + Clone + Copy + Zero,
        C: CancelProgress,
    {
        use itertools::EitherOrBoth::{Both, Left, Right};

        let mut arr1 = Array1::<O>::zeros((self.rows(),));
        let mut arr2 = Array1::<O>::zeros((self.rows(),));

        // only care about every 10000th of the total iterations
        let progress_precision = 10000;

        if self.storage == sprs::CSR {
            let rows = self.rows();
            let progress_modulo = (rows / progress_precision).max(1);

            for (row, vec) in self.data.iter().enumerate() {
                if row % progress_modulo == 0 {
                    snoop.set_progress_check(row as f64 / rows as f64)?;
                }

                // setup coordinating iterators
                let mut iter = cols1
                    .iter()
                    .copied()
                    .merge_join_by(cols2.iter().copied(), std::cmp::Ord::cmp);
                let mut next = iter.next();
                vec.foreach(|col, value| loop {
                    match next {
                        Some(Both(c, _)) | Some(Left(c)) | Some(Right(c)) if c < col => {
                            next = iter.next();
                        }
                        Some(Both(c, _)) if c == col => {
                            let mvalue: O = self.matrix_map.map(value, row, col).into();
                            arr1[row] += mvalue;
                            arr2[row] += mvalue;
                            next = iter.next()
                        }
                        Some(Left(c)) if c == col => {
                            arr1[row] += self.matrix_map.map(value, row, col).into();
                            next = iter.next()
                        }
                        Some(Right(c)) if c == col => {
                            arr2[row] += self.matrix_map.map(value, row, col).into();
                            next = iter.next()
                        }
                        _ => return,
                    }
                })
            }
        } else {
            let cols = cols1.len() + cols2.len();
            let progress_modulo = (cols / progress_precision).max(1);

            let iter = cols1
                .iter()
                .copied()
                .merge_join_by(cols2.iter().copied(), std::cmp::Ord::cmp);

            for (i, step) in iter.enumerate() {
                if i % progress_modulo == 0 {
                    snoop.set_progress_check(i as f64 / cols as f64)?;
                }

                match step {
                    Both(col, _) => {
                        self.data[col].foreach(|row, value| {
                            let mvalue: O = self.matrix_map.map(value, row, col).into();
                            arr1[row] += mvalue;
                            arr2[row] += mvalue;
                        });
                    }
                    Left(col) => {
                        self.data[col].foreach(|row, value| {
                            arr1[row] += self.matrix_map.map(value, row, col).into();
                        });
                    }
                    Right(col) => {
                        self.data[col].foreach(|row, value| {
                            arr2[row] += self.matrix_map.map(value, row, col).into();
                        });
                    }
                }
            }
        }

        snoop.set_progress(1.0);
        Ok((arr1, arr2))
    }

    /// Create an AdaptiveMat from a dense array.
    pub fn from_dense<T: Clone + Into<u32>>(v: ndarray::ArrayView2<T>) -> AdaptiveMat {
        let mut values: Vec<u32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut vecs = Vec::new();
        let (m, n) = v.dim();

        for row in 0..m {
            values.clear();
            indices.clear();

            for col in 0..n {
                let item: u32 = v[(row, col)].clone().into();
                if item > 0 {
                    values.push(item);
                    indices.push(col as u32);
                }
            }

            let vec = AdaptiveVec::new(n, &values, &indices);
            vecs.push(vec);
        }

        AdaptiveMat::new(m, n, sprs::CompressedStorage::CSR, vecs)
    }

    /// sum_cols_diff
    pub fn sum_cols_diff(
        &self,
        cols_a: &[usize],
        cols_b: &[usize],
        norm_factors: Option<&[f64]>,
    ) -> (Vec<f64>, Vec<f64>) {
        let tick = Instant::now();
        let sz = self.rows();

        let mut _cols_a = vec![0u32; self.cols()];
        let mut _cols_b = vec![0u32; self.cols()];
        for idx in cols_a.iter() {
            _cols_a[*idx] = 1u32
        }
        for idx in cols_b.iter() {
            _cols_b[*idx] = 1u32
        }
        info!(
            "sum_cols_diff count_a  {}  count_b {} took {:.3}s",
            cols_a.len(),
            cols_b.len(),
            tick.elapsed().as_millis() as f64 / 1000.0
        );
        if self.storage == sprs::CSR {
            let mut _cols_a = vec![0f64; self.cols()];
            let mut _cols_b = vec![0f64; self.cols()];
            for idx in cols_a.iter() {
                _cols_a[*idx] = 1f64
            }
            for idx in cols_b.iter() {
                _cols_b[*idx] = 1f64
            }
            info!(
                "sum_cols_diff count_a  {}  count_b {} took {:.3}s",
                cols_a.len(),
                cols_b.len(),
                tick.elapsed().as_millis() as f64 / 1000.0
            );
            let ret = match norm_factors {
                None => {
                    let vecs = self.data.iter().collect::<Vec<&AdaptiveVec>>();
                    vecs.par_iter()
                        .map(|&vec| {
                            (
                                vec.iter().fold(0f64, |acc, (col, v)| acc + (v as f64) * _cols_a[col]),
                                vec.iter().fold(0f64, |acc, (col, v)| acc + (v as f64) * _cols_b[col]),
                            )
                        })
                        .collect::<Vec<(f64, f64)>>()
                }
                Some(factors) => self
                    .data
                    .iter()
                    .enumerate()
                    .map(|(_, vec)| {
                        (
                            vec.iter()
                                .fold(0f64, |acc, (col, v)| (acc + (v as f64) * _cols_a[col]) / factors[col]),
                            vec.iter()
                                .fold(0f64, |acc, (col, v)| (acc + (v as f64) * _cols_b[col]) / factors[col]),
                        )
                    })
                    .collect::<Vec<(f64, f64)>>(),
            };
            info!(
                "sum_cols_diff count_a  {}  count_b {} took {:.3}s",
                cols_a.len(),
                cols_b.len(),
                tick.elapsed().as_millis() as f64 / 1000.0
            );
            let arr_a = ret.par_iter().map(|&val| val.0).collect::<Vec<f64>>();
            let arr_b = ret.par_iter().map(|&val| val.1).collect::<Vec<f64>>();
            (arr_a, arr_b)
        } else {
            let mut arr_a = vec![0u32; sz];
            let mut arr_b = vec![0u32; sz];
            match norm_factors {
                None => {
                    for idx in cols_a.iter() {
                        self.data[*idx].foreach(|row, value| arr_a[row] += value);
                    }

                    for idx in cols_b.iter() {
                        self.data[*idx].foreach(|row, value| arr_b[row] += value);
                    }
                }
                Some(factors) => {
                    for idx in cols_a.iter() {
                        self.data[*idx].foreach(|row, value| arr_a[row] += value / (factors[row] as u32));
                    }

                    for idx in cols_b.iter() {
                        self.data[*idx].foreach(|row, value| arr_b[row] += value / (factors[row] as u32));
                    }
                }
            }

            info!(
                "sum_cols_diff count_a  {}  count_b {} took {:.3}s",
                cols_a.len(),
                cols_b.len(),
                tick.elapsed().as_millis() as f64 / 1000.0
            );
            let arr_a = arr_a.par_iter().map(|&val| val as f64).collect::<Vec<f64>>();
            let arr_b = arr_b.par_iter().map(|&val| val as f64).collect::<Vec<f64>>();
            (arr_a, arr_b)
        }
    }

    /// Get the matrix map of this matrix
    pub fn get_map(&self) -> &M {
        &self.matrix_map
    }

    fn sum_axis_exclude(
        &self,
        axis: Axis,
        exclude_rows: &HashSet<usize>,
        exclude_cols: &HashSet<usize>,
    ) -> Array1<f64> {
        let sz = if axis.index() == 0 { self.cols() } else { self.rows() };
        let idx = |row, col| if axis.index() == 0 { col } else { row };
        let mut arr = Array1::zeros((sz,));

        if self.storage == sprs::CSR {
            for (row, vec) in self.data.iter().enumerate() {
                if !exclude_rows.contains(&row) {
                    for (col, value) in vec.iter() {
                        if !exclude_cols.contains(&col) {
                            arr[idx(row, col)] += self.matrix_map.map(value, row, col).into()
                        }
                    }
                }
            }
        } else {
            for (col, vec) in self.data.iter().enumerate() {
                if !exclude_cols.contains(&col) {
                    for (row, value) in vec.iter() {
                        if !exclude_rows.contains(&row) {
                            arr[idx(row, col)] += self.matrix_map.map(value, row, col).into();
                        }
                    }
                }
            }
        }

        arr
    }

    /// Return a filtered matrix has rows and columns that sum to at least some fixed threshold,
    /// along with the omitted columns in a separate matrix, the selected rows, and the selected columns
    pub fn partition_on_threshold(self, threshold: f64) -> AdaptiveMatPartition<N, M> {
        self.partition_on_thresholds(Some(threshold), Some(threshold))
    }

    /// Return a filtered matrix has rows and columns that sum to at least some fixed (possibly unequal)
    /// thresholds, along with the omitted columns in a separate matrix, the selected rows, and the selected columns
    pub fn partition_on_thresholds(
        self,
        row_threshold: Option<f64>,
        col_threshold: Option<f64>,
    ) -> AdaptiveMatPartition<N, M> {
        let mut exclude_rows: HashSet<usize> = HashSet::default();
        let mut exclude_cols: HashSet<usize> = HashSet::default();

        // repeat removing rows and cols below threshold until we stabilize
        loop {
            let mut updated = false;
            if let Some(threshold) = col_threshold {
                let sum = self.sum_axis_exclude(Axis(0), &exclude_rows, &exclude_cols);
                for (col, &s) in sum.iter().enumerate() {
                    if s < threshold {
                        updated |= exclude_cols.insert(col);
                    }
                }
            }
            if let Some(threshold) = row_threshold {
                let sum = self.sum_axis_exclude(Axis(1), &exclude_rows, &exclude_cols);
                for (row, &s) in sum.iter().enumerate() {
                    if s < threshold {
                        updated |= exclude_rows.insert(row);
                    }
                }
            }
            if !updated {
                break;
            }
        }

        let nrows = self.rows - exclude_rows.len();
        let fcols = self.cols - exclude_cols.len();
        let rcols = exclude_cols.len();

        let (filtered_data, residual_data) = {
            let mut f = vec![];
            let mut r = vec![];

            if self.storage == sprs::CSR {
                let column_map = {
                    let mut m = vec![0; self.cols];
                    let mut nexcl = 0;
                    let mut nincl = 0;
                    for (col, m) in m.iter_mut().enumerate() {
                        if exclude_cols.contains(&col) {
                            *m = nexcl;
                            nexcl += 1;
                        } else {
                            *m = nincl;
                            nincl += 1;
                        }
                    }
                    m
                };
                for (row, vec) in self.data.iter().enumerate() {
                    let mut fvals = Vec::new();
                    let mut fidxs = Vec::new();
                    let mut rvals = Vec::new();
                    let mut ridxs = Vec::new();
                    for (col, val) in vec.iter() {
                        if exclude_cols.contains(&col) {
                            rvals.push(val);
                            ridxs.push(column_map[col]);
                        } else {
                            fvals.push(val);
                            fidxs.push(column_map[col]);
                        }
                    }
                    let fvec = AdaptiveVec::new(fcols, &fvals, &fidxs);
                    let rvec = AdaptiveVec::new(rcols, &rvals, &ridxs);
                    if !exclude_rows.contains(&row) {
                        f.push(fvec);
                        r.push(rvec);
                    }
                }
            } else {
                let row_map = {
                    let mut m = vec![0; self.rows];
                    let mut nincl = 0;
                    for (row, m) in m.iter_mut().enumerate() {
                        if !exclude_rows.contains(&row) {
                            *m = nincl;
                            nincl += 1;
                        }
                    }
                    m
                };
                for (col, vec) in self.data.iter().enumerate() {
                    let mut nvals = Vec::new();
                    let mut nidxs = Vec::new();
                    for (row, val) in vec.iter() {
                        if !exclude_rows.contains(&row) {
                            nvals.push(val);
                            nidxs.push(row_map[row]);
                        }
                    }
                    let nvec = AdaptiveVec::new(nrows, &nvals, &nidxs);
                    if exclude_cols.contains(&col) {
                        r.push(nvec);
                    } else {
                        f.push(nvec);
                    }
                }
            }

            (f, r)
        };

        let filtered = AdaptiveMat::new_with_map(nrows, fcols, self.storage, filtered_data, self.matrix_map.clone());
        let residuals = AdaptiveMat::new_with_map(nrows, rcols, self.storage, residual_data, self.matrix_map.clone());

        let selected_rows = (0..self.rows).filter(|r| !exclude_rows.contains(r)).collect::<Vec<_>>();
        let selected_cols = (0..self.cols).filter(|c| !exclude_cols.contains(c)).collect::<Vec<_>>();

        (filtered, residuals, selected_rows, selected_cols)
    }

    /// Replace the `MatrixMap` with a new `MatrixMap`, potentially having a new output type
    pub fn set_map<NewMap, NewN>(self, new_map: NewMap) -> AdaptiveMat<NewN, D, NewMap>
    where
        NewN: AdaptiveMatNum,
        NewMap: MatrixMap<u32, NewN>,
    {
        AdaptiveMat::new_with_map(self.rows, self.cols, self.storage, self.data, new_map)
    }

    /// Compose the original `MatrixMap` with a new `MatrixMap` and replace the original
    pub fn compose_map<NewMapIn, NewMapOut, NewMap>(
        self,
        new_map: NewMap,
    ) -> AdaptiveMat<NewMapOut, D, impl MatrixMap<u32, NewMapOut>>
    where
        NewMapIn: Clone,
        N: Into<NewMapIn>,
        NewMapOut: AdaptiveMatNum,
        NewMap: MatrixMap<NewMapIn, NewMapOut>,
    {
        let composed_map = ComposedMap::new(self.matrix_map.clone(), new_map);
        self.set_map(composed_map)
    }

    /// Convert the output type `N` to `NewN` by composing with a new `MatrixMap` (`MatrixIntoMap<N, NewN>`)
    pub fn values_into<NewN>(self) -> AdaptiveMat<NewN, D, impl MatrixMap<u32, NewN>>
    where
        NewN: AdaptiveMatNum,
        N: Into<NewN>,
    {
        self.compose_map(MatrixIntoMap::<N, NewN>::new())
    }

    /// Apply a function to all elements by composing with any existing `MatrixMap`
    pub fn apply<FIn, FOut, F>(self, f: F) -> AdaptiveMat<FOut, D, impl MatrixMap<u32, FOut>>
    where
        FIn: Copy + Num,
        N: Into<FIn>,
        FOut: AdaptiveMatNum,
        F: Clone + Fn(FIn) -> FOut,
    {
        self.compose_map(ScalarMap::new(f))
    }

    /// Center the columns (assuming `Axis(0)`) to have mean 0. The column means can optionally
    /// be provided with `m`, in which case each row has `m` subtracted element-wise.
    pub fn center(self, axis: Axis, m: Option<Array1<f64>>) -> LowRankOffset<D, impl MatrixMap<u32, f64>> {
        assert!(axis.index() < 2);

        let neg_means = match m {
            Some(means) => means,
            None => self.mean_axis(axis),
        }
        .mapv_into(|x| -x);

        match axis {
            // Centering the columns
            Axis(0) => {
                let u = Array::ones((self.rows(), 1));
                let v = neg_means.into_shape((1, self.cols())).unwrap();
                // self.values_into() because LowRankOffset requires f64
                LowRankOffset::new(self.values_into(), u, v)
            }
            // Centering the rows
            Axis(1) => {
                let u = neg_means.into_shape((self.rows(), 1)).unwrap();
                let v = Array::ones((1, self.cols()));
                LowRankOffset::new(self.values_into(), u, v)
            }
            _ => unreachable!(),
        }
    }

    /// Scale the columns (assuming `Axis(0)`) to have variance 1. The column std. deviations
    /// can optionally be provided with `s`, in which case each row is divided element-wise by `s`.
    pub fn scale(self, axis: Axis, s: Option<Array1<f64>>) -> AdaptiveMat<f64, D, impl MatrixMap<u32, f64>> {
        assert!(axis.index() < 2);

        let scale_factors = match s {
            Some(std_devs) => std_devs.mapv_into(|x| 1.0 / x),
            None => {
                let means_sq = self.mean_axis(axis).mapv_into(|x| x.powi(2));
                let sq_means = self.view().apply(|x| x.powi(2)).mean_axis(axis);
                (sq_means - means_sq).mapv_into(|x| if x == 0.0 { 1.0 } else { 1.0 / x.sqrt() })
            }
        };

        // if summing along rows, we are scaling columns, and vice versa
        let axis_to_scale = Axis(1 - axis.index());
        self.compose_map(ScaleAxis::new(axis_to_scale, scale_factors))
    }

    /// Scale and center each slice of the matrix along `axis` to have variance 1 and mean 0
    /// `Axis(0)` applies to columns; `Axis(1)` applies to rows.
    /// Panics if variance of any slice is 0.
    pub fn scale_and_center(
        self,
        axis: Axis,
        scaling_factors: Option<Array1<f64>>,
    ) -> LowRankOffset<D, impl MatrixMap<u32, f64>> {
        assert!(axis.index() < 2);

        let mut means = self.mean_axis(axis);
        let scaling_factors = scaling_factors.unwrap_or_else(|| {
            let matsq_means = self.view().apply(|x| x.powi(2)).mean_axis(axis);
            (matsq_means - means.mapv(|x| x.powi(2))).mapv_into(|x| if x <= 0.0 { 1.0 } else { x.sqrt() })
        });
        means /= &scaling_factors;

        self.scale(axis, Some(scaling_factors)).center(axis, Some(means))
    }

    /// Select the columns of interest
    pub fn select_cols(&self, selected_cols: &[usize]) -> AdaptiveMatOwned<N, M> {
        let mut data = Vec::new();
        if self.storage == sprs::CSR {
            data.reserve_exact(self.data.len());
            let mut tmp = Vec::new();
            for (_row, vec) in self.data.iter().enumerate() {
                let mut nvals = Vec::new();
                let mut nidxs = Vec::new();
                vec.to_vec(&mut tmp);
                for (new_col, &col) in selected_cols.iter().enumerate() {
                    let val = tmp[col];
                    if val > 0 {
                        nvals.push(val);
                        nidxs.push(new_col as u32);
                    }
                }
                let nvec = AdaptiveVec::new(selected_cols.len(), &nvals, &nidxs);
                data.push(nvec);
            }
        } else {
            data.reserve_exact(selected_cols.len());
            for &col in selected_cols {
                data.push(self.data[col].clone());
            }
        }
        AdaptiveMatOwned {
            rows: self.rows,
            cols: selected_cols.len(),
            storage: self.storage,
            data,
            matrix_map: self.matrix_map.clone(),
            _n: self._n,
        }
    }

    /// Select the rows of interest
    pub fn select_rows(&self, selected_rows: &[usize]) -> AdaptiveMatOwned<N, M> {
        let mut data = vec![];
        if self.storage == sprs::CSR {
            for &row in selected_rows {
                data.push(self.data[row].clone());
            }
        } else {
            let mut tmp = vec![];
            for (_col, vec) in self.data.iter().enumerate() {
                let mut nvals = Vec::new();
                let mut nidxs = Vec::new();
                vec.to_vec(&mut tmp);
                for (new_row, &row) in selected_rows.iter().enumerate() {
                    let val = tmp[row];
                    if val > 0 {
                        nvals.push(val);
                        nidxs.push(new_row as u32);
                    }
                }
                let nvec = AdaptiveVec::new(selected_rows.len(), &nvals, &nidxs);
                data.push(nvec);
            }
        }
        AdaptiveMatOwned {
            rows: selected_rows.len(),
            cols: self.cols,
            storage: self.storage,
            data,
            matrix_map: self.matrix_map.clone(),
            _n: self._n,
        }
    }
}

impl<'a, 'b, N, D, M, A, DS> Dot<ArrayBase<DS, Ix2>> for AdaptiveMat<N, D, M>
where
    N: AdaptiveMatNum,
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, N>,
    A: 'a + Copy + Default + Num + Mul<N, Output = A>,
    DS: 'b + ndarray::Data<Elem = A>,
{
    type Output = Array<A, Ix2>;

    fn dot(&self, rhs: &ArrayBase<DS, Ix2>) -> Self::Output {
        let mut out = Array2::zeros((self.rows(), rhs.shape()[1]));

        prod::mat_densemat_mult(self, rhs, out.view_mut());
        out
    }
}

impl<'a, 'b, N, D, M, A, DS> Dot<ArrayBase<DS, Ix1>> for AdaptiveMat<N, D, M>
where
    N: AdaptiveMatNum,
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, N>,
    A: 'a + Copy + Default + Num + Mul<N, Output = A>,
    DS: 'b + ndarray::Data<Elem = A>,
{
    type Output = Array<A, Ix1>;

    fn dot(&self, rhs: &ArrayBase<DS, Ix1>) -> Self::Output {
        // reshape the vector into a mx1 matrix
        // FIXME this is a slight hack
        let rhs = ndarray::ArrayView2::from_shape((rhs.shape()[0], 1), rhs.as_slice().unwrap()).unwrap();

        let out = self.dot(&rhs);

        // reshape the output to 1d
        out.into_shape(self.rows()).unwrap()
    }
}

impl<'a, 'b, N, D, M, A, DS> Dot<AdaptiveMat<N, D, M>> for ArrayBase<DS, Ix2>
where
    N: AdaptiveMatNum,
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, N>,
    A: 'a + Copy + Default + Num + Mul<N, Output = A>,
    DS: 'b + ndarray::Data<Elem = A>,
{
    type Output = Array<A, Ix2>;

    fn dot(&self, rhs: &AdaptiveMat<N, D, M>) -> Self::Output {
        let lhs_t = self.t();
        let rhs_t = rhs.t();

        let mut out = Array2::zeros((rhs_t.rows(), lhs_t.shape()[1]));

        prod::mat_densemat_mult(&rhs_t, &lhs_t, out.view_mut());
        out.reversed_axes()
    }
}

impl<'a, 'b, N, D, M, A, DS> Dot<&AdaptiveMat<N, D, M>> for ArrayBase<DS, Ix2>
where
    N: AdaptiveMatNum,
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, N>,
    A: 'a + Copy + Default + Num + Mul<N, Output = A>,
    DS: 'b + ndarray::Data<Elem = A>,
{
    type Output = Array<A, Ix2>;

    fn dot(&self, rhs: &&AdaptiveMat<N, D, M>) -> Self::Output {
        self.dot(*rhs)
    }
}

impl<'a, 'b, N, D, M, A, DS> Dot<AdaptiveMat<N, D, M>> for ArrayBase<DS, Ix1>
where
    N: AdaptiveMatNum,
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, N>,
    A: 'a + Copy + Default + Num + Mul<N, Output = A>,
    DS: 'b + ndarray::Data<Elem = A>,
{
    type Output = Array<A, Ix1>;

    fn dot(&self, rhs: &AdaptiveMat<N, D, M>) -> Self::Output {
        // reshape the vector into a mx1 matrix
        // FIXME this is a slight hack
        let lhs = ndarray::ArrayView2::from_shape((1, self.shape()[0]), self.as_slice().unwrap()).unwrap();

        let out = lhs.dot(rhs);

        // reshape the output to 1d
        out.into_shape(rhs.cols()).unwrap()
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::gen_rand::{random_adaptive_mat, random_dense_mat};
    use crate::TransposeMap;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, s, ArrayView, Dimension};
    use rand::{
        distributions::Uniform,
        prelude::{Rng, SeedableRng},
    };
    use rand_pcg::Pcg64Mcg;

    #[derive(Clone, Debug, PartialEq)]
    enum RandomMap {
        OffsetAndLog,
        Scale(f64),
        Ramp(f64, f64),
    }

    impl Eq for RandomMap {}

    impl MatrixMap<u32, f64> for RandomMap {
        type T = TransposeMap<u32, f64, Self>;

        #[inline(always)]
        fn map(&self, u: u32, r: usize, c: usize) -> f64 {
            match self {
                RandomMap::OffsetAndLog => ((u + 1) as f64).ln(),
                RandomMap::Scale(s) => u as f64 * s,
                RandomMap::Ramp(s1, s2) => u as f64 * s1 * r as f64 * s2 * c as f64,
            }
        }

        fn t(&self) -> Self::T {
            TransposeMap::new(self.clone())
        }
    }

    impl RandomMap {
        fn gen_random(r: &mut impl Rng) -> RandomMap {
            let which = r.gen_range(0..3);
            if which == 0 {
                RandomMap::OffsetAndLog
            } else if which == 1 {
                RandomMap::Scale(r.gen_range(0.0001..100.0))
            } else if which == 2 {
                RandomMap::Ramp(r.gen_range(0.00001..0.01), r.gen_range(0.00001..0.01))
            } else {
                unreachable!()
            }
        }
    }

    pub fn random_matrices(n: usize, step: usize) -> impl Iterator<Item = (usize, usize, AdaptiveMat)> {
        let mut rng = Pcg64Mcg::seed_from_u64(42);

        (0..n).step_by(step).map(move |i| {
            let rows = rng.gen_range(0..i + 1) + rng.gen_range(0..i + 1);
            let cols = rng.gen_range(0..i + 1) + rng.gen_range(0..i + 1);

            let range = rng.gen_range(2..50);
            (rows, cols, random_adaptive_mat(&mut rng, rows, cols, range, None))
        })
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
    /// Test sum_cols
    fn test_misc() {
        let arr_u32: Array2<u32> = array![[1, 2, 3], [2, 3, 4], [3, 4, 5]];
        let mat = AdaptiveMatOwned::<f64>::from_dense(arr_u32.view());
        let dense = arr_u32.mapv(|x| x as f64);

        for idx in 0..2 {
            let axis = Axis(idx);

            let dense_sum = dense.sum_axis(axis);
            let mat_sum = mat.sum_axis(axis);
            assert_close(dense_sum.view(), mat_sum.view());

            let dense_mean = dense.mean_axis(axis).unwrap();
            let mat_mean = mat.mean_axis(axis);
            assert_close(dense_mean.view(), mat_mean.view());

            let dense_var = dense.var_axis(axis, 0.0);
            let mat_var = mat.var_axis(axis);
            assert_close(dense_var.view(), mat_var.view());
        }

        let centered_cols = array![[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        assert_close(centered_cols.view(), mat.view().center(Axis(0), None).to_dense().view());

        let centered_rows = array![[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
        assert_close(centered_rows.view(), mat.view().center(Axis(1), None).to_dense().view());

        println!("OK");
        println!("{:?}", mat.view().scale_and_center(Axis(0), None).to_dense());
        println!("{:?}", mat.view().scale_and_center(Axis(1), None).to_dense());
    }

    fn input_a() -> Array2<u32> {
        array![
            [136, 936, 0, 0, 264],
            [134, 682, 417, 8, 391],
            [0, 133, 780, 885, 0],
            [396, 76, 96, 198, 0],
        ]
    }

    #[test]
    /// Test sum fns
    fn test_sum_fns() {
        let dense = input_a();
        let sparse = AdaptiveMatOwned::<u32>::from_dense(dense.view());
        let sum0 = sparse.sum_axis::<u32>(Axis(0));
        let e_sum0 = array![666, 1827, 1293, 1091, 655];
        assert!(e_sum0 == sum0);
        let sum1 = sparse.sum_axis::<u32>(Axis(1));
        let e_sum1 = array![1336, 1632, 1798, 766];
        assert!(e_sum1 == sum1);
        let cols1 = vec![1, 2, 3];
        let sumc = sparse.sum_cols::<u32>(&cols1);
        let e_sumc = dense.slice(s![.., 1..=3]).sum_axis(Axis(0));
        assert!(e_sumc == sumc);
        let sumr = sparse.sum_rows::<u32>(&cols1);
        let e_sumr1 = dense.slice(s![.., 1..=3]).sum_axis(Axis(1));
        assert!(e_sumr1 == sumr);
        let cols2 = vec![2, 3, 4];
        let (sumr1, sumr2) = sparse.sum_rows_dual::<u32>(&cols1, &cols2);
        let e_sumr2 = dense.slice(s![.., 2..=4]).sum_axis(Axis(1));
        assert!(e_sumr1 == sumr1);
        assert!(e_sumr2 == sumr2);
    }

    #[test]
    /// Test mean fns
    fn test_mean_fns() {
        let dense = input_a();
        let sparse = AdaptiveMatOwned::<u32>::from_dense(dense.view());
        let mean0 = sparse.mean_axis(Axis(0));
        let e_mean0 = array![166.5, 456.75, 323.25, 272.75, 163.75];
        assert_abs_diff_eq!(e_mean0, mean0, epsilon = 1e-7);
        let mean1 = sparse.mean_axis(Axis(1));
        let e_mean1 = array![267.2, 326.4, 359.6, 153.2];
        assert_abs_diff_eq!(e_mean1, mean1, epsilon = 1e-7);
        let cols = vec![1, 2, 3];
        let meanc = sparse.mean_rows(&cols);
        let e_meanc = dense
            .slice(s![.., 1..=3])
            .mapv(|v| v as f64)
            .mean_axis(Axis(1))
            .unwrap();
        assert_abs_diff_eq!(e_meanc, meanc, epsilon = 1e-7);
    }

    #[test]
    /// Test mean+var fns
    fn test_mean_var_fns() {
        let dense = input_a();
        let sparse = AdaptiveMatOwned::<u32>::from_dense(dense.view());
        let (mean0, var0) = sparse.mean_var_axis(Axis(0));
        let e_mean0 = array![166.5, 456.75, 323.25, 272.75, 163.75];
        let e_var0 = array![20594.75, 132550.6875, 93385.6875, 131230.6875, 28830.1875];
        assert_abs_diff_eq!(e_mean0, mean0, epsilon = 1e-7);
        assert_abs_diff_eq!(e_var0, var0, epsilon = 1e-7);
        let (mean1, var1) = sparse.mean_var_axis(Axis(1));
        let e_mean1 = array![267.2, 326.4, 359.6, 153.2];
        let e_var1 = array![121461.76, 55445.84, 152550.64, 18732.16];
        assert_abs_diff_eq!(e_mean1, mean1, epsilon = 1e-7);
        assert_abs_diff_eq!(e_var1, var1, epsilon = 1e-7);
        let cols = vec![1, 2, 3];
        let (meanc, varc) = sparse.mean_var_rows(&cols);
        let densef = dense.slice(s![.., 1..=3]).mapv(|v| v as f64);
        let e_meanc = densef.mean_axis(Axis(1)).unwrap();
        let e_varc = densef.var_axis(Axis(1), 0.0);
        assert_abs_diff_eq!(e_meanc, meanc, epsilon = 1e-7);
        assert_abs_diff_eq!(e_varc, varc, epsilon = 1e-7);
    }

    /// Test that we can round-trip from AdaptiveMat to CsMat and back.
    #[test]
    fn sprs_conversion_simple() {
        for (_, _, sparse) in random_matrices(1000, 10) {
            let csmat = sparse.to_csmat();
            let round_trip = AdaptiveMatOwned::from_csmat(&csmat);
            assert_eq!(sparse, round_trip);
        }
    }

    /// Test matrix editing is self-consistent
    #[test]
    fn sprs_conversion_map() {
        let rng = &mut Pcg64Mcg::seed_from_u64(42);

        for (_, _, orig) in random_matrices(1000, 10) {
            let rand_edit = RandomMap::gen_random(rng);

            let orig_dense = orig.to_dense();
            let orig_edit = orig.set_map(rand_edit.clone());
            let orig_edit_dense = orig_edit.to_dense();
            let orig_edit_sparse_dense = orig_edit.to_csmat().to_dense();

            let mut orig_dense_edit = Array2::zeros(orig_dense.dim());

            for (idx, v) in orig_dense.indexed_iter() {
                orig_dense_edit[idx] = rand_edit.map(*v, idx.0, idx.1);
            }

            assert_eq!(orig_edit_dense, orig_dense_edit);
            assert_eq!(orig_edit_sparse_dense, orig_dense_edit);
        }
    }

    #[test]
    fn test_sparse_dot_dense() {
        let rng = &mut Pcg64Mcg::seed_from_u64(42);

        for (_rows, cols, sparse) in random_matrices(2000, 30) {
            let sparse_as_dense = sparse.to_dense();

            // Generate a random dense 'query' matrix
            let dense_cols = rng.gen_range(0..64);
            let dense = random_dense_mat(rng, cols, dense_cols);

            // correct answer
            let truth = sparse_as_dense.dot(&dense);

            // test our math
            let test = sparse.dot(&dense);

            assert_eq!(truth, test);
        }
    }

    #[test]
    fn test_sparse_dot_dense_1d() {
        let rng = &mut Pcg64Mcg::seed_from_u64(42);

        for (_rows, cols, sparse) in random_matrices(2000, 100) {
            let sparse_as_dense = sparse.to_dense();

            // Generate a random dense 'query' vector
            let dense = random_dense_mat(rng, cols, 1).into_shape(cols).unwrap();

            // correct answer
            let truth = sparse_as_dense.dot(&dense);

            // test our math
            let test = sparse.dot(&dense);

            assert_eq!(truth, test);
        }
    }

    #[test]
    fn test_dense_dot_sparse() {
        let rng = &mut Pcg64Mcg::seed_from_u64(42);

        for (rows, _cols, sparse) in random_matrices(2000, 100) {
            let sparse_as_dense = sparse.to_dense();

            // Generate a random dense 'query' matrix
            let dense_rows = rng.gen_range(0..64);
            let dense = random_dense_mat(rng, dense_rows, rows);

            // correct answer
            let truth = dense.dot(&sparse_as_dense);

            // test our math
            let test = dense.dot(&sparse);

            assert_eq!(truth, test);
        }
    }

    #[test]
    fn test_dense_dot_sparse_1d() {
        let rng = &mut Pcg64Mcg::seed_from_u64(42);

        for (rows, _cols, sparse) in random_matrices(2000, 50) {
            let sparse_as_dense = sparse.to_dense();

            // Generate a random dense 'query' matrix
            let dense = random_dense_mat(rng, 1, rows).into_shape(rows).unwrap();

            // correct answer
            let truth = dense.dot(&sparse_as_dense);

            // test our math
            let test = dense.dot(&sparse);

            assert_eq!(truth, test);
        }
    }

    #[test]
    fn test_partition_on_thresholds() {
        use ndarray_stats::{interpolate::Midpoint, Quantile1dExt};
        use noisy_float::types::N64;

        for (rows, cols, sparse) in random_matrices(2000, 50) {
            // skip over the smaller stuff for now
            if rows.min(cols) < 100 {
                continue;
            }

            let dense = sparse.to_dense();

            let mut sum0 = dense.sum_axis(Axis(0));
            let col_threshold = sum0.quantile_mut(N64::new(0.1), &Midpoint {}).unwrap();
            let mut sum1 = dense.sum_axis(Axis(1));
            let row_threshold = sum1.quantile_mut(N64::new(0.1), &Midpoint {}).unwrap();

            let sparse_shape = sparse.shape();
            let (filt, res, sel_rows, sel_cols) =
                sparse.partition_on_thresholds(Some(row_threshold as f64), Some(col_threshold as f64));

            let (filt_dense, res_dense, sel_rows_dense, sel_cols_dense) = {
                let mut excl_rows: HashSet<usize> = HashSet::default();
                let mut excl_cols: HashSet<usize> = HashSet::default();

                loop {
                    let mut updated = false;
                    let mut sum = Array1::<u32>::zeros((cols,));
                    for (idx, row) in dense.axis_iter(Axis(0)).enumerate() {
                        if !excl_rows.contains(&idx) {
                            sum += &row;
                        }
                    }
                    for (col, &s) in sum.iter().enumerate() {
                        if s < col_threshold {
                            updated |= excl_cols.insert(col);
                        }
                    }
                    let mut sum = Array1::<u32>::zeros((rows,));
                    for (idx, col) in dense.axis_iter(Axis(1)).enumerate() {
                        if !excl_cols.contains(&idx) {
                            sum += &col;
                        }
                    }
                    for (row, &s) in sum.iter().enumerate() {
                        if s < row_threshold {
                            updated |= excl_rows.insert(row);
                        }
                    }
                    if !updated {
                        break;
                    }
                }

                let incl_rows = (0..rows).filter(|row| !excl_rows.contains(row)).collect::<Vec<_>>();
                let incl_cols = (0..cols).filter(|col| !excl_cols.contains(col)).collect::<Vec<_>>();
                let mut excl_cols = excl_cols.into_iter().collect::<Vec<_>>();
                excl_cols.sort_unstable();

                let filt = dense.select(Axis(0), &incl_rows);
                let res = filt.select(Axis(1), &excl_cols);
                let filt = filt.select(Axis(1), &incl_cols);

                (filt, res, incl_rows, incl_cols)
            };

            assert_ne!(sparse_shape, filt.shape());
            assert_eq!(filt_dense, filt.to_dense());
            assert_eq!(res_dense, res.to_dense());
            assert_eq!(sel_rows_dense, sel_rows);
            assert_eq!(sel_cols_dense, sel_cols);
        }
    }

    #[test]
    fn test_select() {
        let rng = &mut Pcg64Mcg::seed_from_u64(42);

        for (nrows, ncols, sparse) in random_matrices(2000, 100) {
            if nrows == 0 || ncols == 0 {
                continue;
            }
            let sparse_as_dense = sparse.to_dense();

            let rows = rng.sample_iter(Uniform::new(0, nrows)).take(100).collect::<Vec<_>>();
            let selected = sparse.select_rows(&rows).to_dense();
            for (i, &row) in rows.iter().enumerate() {
                assert_eq!(selected.slice(s![i, ..]), sparse_as_dense.slice(s![row, ..]));
            }

            let cols = rng.sample_iter(Uniform::new(0, ncols)).take(100).collect::<Vec<_>>();
            let selected = sparse.select_cols(&cols).to_dense();
            for (j, &col) in cols.iter().enumerate() {
                assert_eq!(selected.slice(s![.., j]), sparse_as_dense.slice(s![.., col]));
            }
        }
    }
}
