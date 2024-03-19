use crate::mat::AdaptiveMatNum;
use crate::{ada_expand, AbstractVec, AdaptiveMat, AdaptiveVec, MatrixMap};
use multiversion::multiversion;
use ndarray::{ArrayBase, ArrayView, ArrayViewMut, Axis, Ix1, Ix2};
use num_traits::Num;
use std::ops::{Deref, Mul};

/// Multiply the sparse matrix `lhs` by the dense matrix `rhs`, accumulating results into the dense matrix view `out`.
pub fn mat_densemat_mult<'a, 'b, N, D, M, A, DS>(
    lhs: &AdaptiveMat<N, D, M>,
    rhs: &ArrayBase<DS, Ix2>,
    out: ArrayViewMut<'a, A, Ix2>,
) where
    N: AdaptiveMatNum,
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, N>,
    A: 'a + Num + Copy + Mul<N, Output = A> + Clone,
    DS: 'b + ndarray::Data<Elem = A>,
{
    if lhs.storage == sprs::CSR {
        csrmat_densemat_mult(lhs, rhs, out);
    } else {
        cscmat_densemat_mult(lhs, rhs, out);
    }
}

/// CSR-dense rowmaj multiplication
///
/// Performs better if rhs has a decent number of columns.
fn csrmat_densemat_mult<'a, 'b, N, D, M, A, DS>(
    lhs: &AdaptiveMat<N, D, M>,
    rhs: &ArrayBase<DS, Ix2>,
    mut out: ArrayViewMut<'a, A, Ix2>,
) where
    N: AdaptiveMatNum,
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, N>,
    A: 'a + Num + Copy + Mul<N, Output = A> + Clone,
    DS: 'b + ndarray::Data<Elem = A>,
{
    if lhs.cols() != rhs.shape()[0] {
        panic!("Dimension mismatch");
    }
    if lhs.rows() != out.shape()[0] {
        panic!("Dimension mismatch");
    }
    if rhs.shape()[1] != out.shape()[1] {
        panic!("Dimension mismatch");
    }
    if !lhs.is_csr() {
        panic!("Storage mismatch");
    }

    let axis0 = Axis(0);

    for (row, (vec, oline)) in lhs.data.iter().zip(out.axis_iter_mut(axis0)).enumerate() {
        ada_expand!(vec, v, vec_mulacc_dense_rowmaj(v, row, &lhs.matrix_map, rhs, oline));
    }
}

/// CSC-dense rowmaj multiplication
///
/// Performs better if rhs has a decent number of columns.
fn cscmat_densemat_mult<'a, 'b, N, D, M, A, DS>(
    lhs: &AdaptiveMat<N, D, M>,
    rhs: &ArrayBase<DS, Ix2>,
    mut out: ArrayViewMut<'a, A, Ix2>,
) where
    N: AdaptiveMatNum,
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, N>,
    A: 'a + Num + Copy + Mul<N, Output = A> + Clone,
    DS: 'b + ndarray::Data<Elem = A>,
{
    if lhs.cols() != rhs.shape()[0] {
        panic!("Dimension mismatch");
    }
    if lhs.rows() != out.shape()[0] {
        panic!("Dimension mismatch");
    }
    if rhs.shape()[1] != out.shape()[1] {
        panic!("Dimension mismatch");
    }
    if !lhs.is_csc() {
        panic!("Storage mismatch");
    }

    let axis0 = Axis(0);

    for (col, (vec, rhsline)) in lhs.data.iter().zip(rhs.axis_iter(axis0)).enumerate() {
        ada_expand!(
            vec,
            v,
            colvec_mulacc_dense_rowmaj(v, col, &lhs.matrix_map, &rhsline, &mut out)
        );
    }
}

/// CSR-dense rowmaj multiplication
///
/// Performs better if rhs has a decent number of columns.
#[inline(never)]
fn vec_mulacc_dense_rowmaj<'a, 'b, N, V, M, A, DS>(
    lhs: &V,
    vec_row: usize,
    matrix_map: &M,
    rhs: &ArrayBase<DS, Ix2>,
    mut out: ArrayViewMut<'a, A, Ix1>,
) where
    N: Copy,
    V: AbstractVec<Output = u32>,
    M: MatrixMap<u32, N>,
    A: 'a + Num + Copy + Mul<N, Output = A> + Clone,
    DS: 'b + ndarray::Data<Elem = A>,
{
    if lhs.len() != rhs.shape()[0] {
        panic!("Dimension mismatch");
    }

    if rhs.is_standard_layout() && out.is_standard_layout() {
        let rhs_mat = rhs.as_slice().unwrap();
        let out_slice = out.as_slice_mut().unwrap();
        vec_mulacc_dense_rowmaj_std(lhs, vec_row, matrix_map, rhs_mat, out_slice);
        return;
    }

    for (ind, lval) in lhs.iter() {
        let lval = matrix_map.map(lval, vec_row, ind);

        let rline = rhs.row(ind);
        for (oval, &rval) in out.iter_mut().zip(rline.iter()) {
            let prev = *oval;
            *oval = prev + rval * lval;
        }
    }
}

/// CSR-dense rowmaj multiplication
///
/// Performs better if rhs has a decent number of columns.
#[multiversion(targets("x86_64+avx+fma", "x86_64+avx", "x86_64+sse3"))]
fn vec_mulacc_dense_rowmaj_std<'a, 'b, N, V, M, A>(
    lhs: &V,
    vec_row: usize,
    matrix_map: &M,
    rhs_mat: &[A],
    out_slice: &mut [A],
) where
    N: Copy,
    V: AbstractVec<Output = u32>,
    M: MatrixMap<u32, N>,
    A: 'a + Num + Copy + Mul<N, Output = A> + Clone,
{
    let out_cols = out_slice.len();

    for (ind, lval) in lhs.iter() {
        let lval = matrix_map.map(lval, vec_row, ind);

        let rhs_slice = &rhs_mat[out_cols * ind..out_cols * (ind + 1)];

        // this should be vectorized to avx
        for (o, r) in out_slice.iter_mut().zip(rhs_slice) {
            *o = *o + *r * lval;
        }
    }
}

/// CSR-dense rowmaj multiplication
///
/// Performs better if rhs has a decent number of column.
#[inline(never)]
fn colvec_mulacc_dense_rowmaj<'a, 'b, N, V, M, A>(
    lhs: &V,
    lhs_col: usize,
    matrix_map: &M,
    rhs: &ArrayView<A, Ix1>,
    out: &mut ArrayViewMut<'a, A, Ix2>,
) where
    N: Copy,
    V: AbstractVec<Output = u32>,
    M: MatrixMap<u32, N>,
    A: 'a + Num + Copy + Mul<N, Output = A> + Clone,
{
    if lhs.len() != out.shape()[0] {
        panic!("Dimension mismatch");
    }

    if rhs.is_standard_layout() && out.is_standard_layout() {
        let rhs_slice = rhs.as_slice().unwrap();
        let out_slice = out.as_slice_mut().unwrap();
        colvec_mulacc_dense_rowmaj_std(lhs, lhs_col, matrix_map, rhs_slice, out_slice);
        return;
    }

    for (ind, lval) in lhs.iter() {
        let lval = matrix_map.map(lval, ind, lhs_col);

        let mut oline = out.row_mut(ind);
        for (&rval, oval) in rhs.iter().zip(oline.iter_mut()) {
            let prev = *oval;
            *oval = prev + rval * lval;
        }
    }
}

/// CSR-dense rowmaj multiplication
///
/// Performs better if rhs has a decent number of columns.
#[multiversion(targets("x86_64+avx+fma", "x86_64+avx", "x86_64+sse3"))]
fn colvec_mulacc_dense_rowmaj_std<'a, 'b, N, V, M, A>(
    lhs: &V,
    lhs_col: usize,
    matrix_map: &M,
    rhs_row_slice: &[A],
    out_mat: &mut [A],
) where
    N: Copy,
    V: AbstractVec<Output = u32>,
    M: MatrixMap<u32, N>,
    A: 'a + Num + Copy + Mul<N, Output = A> + Clone,
{
    let rhs_cols = rhs_row_slice.len();

    for (ind, lval) in lhs.iter() {
        let lval = matrix_map.map(lval, ind, lhs_col);

        let out_slice = &mut out_mat[rhs_cols * ind..rhs_cols * (ind + 1)];

        // this should be vectorized to avx
        for (o, r) in out_slice.iter_mut().zip(rhs_row_slice) {
            *o = *o + *r * lval;
        }
    }
}
