use ndarray::{s, Array, Array1, Array2, Axis, NdFloat, Zip};
#[cfg(feature = "plotly")]
use plotly::{
    color::Rgb,
    common::{Marker, Mode, Title},
    Layout, Plot, Scatter,
};
use rand::Rng;
use std::fmt::Debug;

pub type Q = f64;

#[cfg(feature = "plotly")]
pub const COLORS: [usize; 10] = [
    0x006400, 0x00008b, 0xb03060, 0xff4500, 0xffd700, 0x7fff00, 0x00ffff, 0xff00ff, 0x6495ed, 0xffdab9,
];

#[derive(Debug, PartialEq, Clone)]
pub struct LabelledVector {
    pub id: u8,
    pub values: Vec<f32>,
}

#[inline]
pub fn dot_prod(a: &[Q], b: &[Q]) -> Q {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0_f64, |acc, (&a, &b)| acc + (a * b))
}

#[inline]
pub fn magnitude(vec: &[Q]) -> Q {
    dot_prod(vec, vec).sqrt()
}

/// Transforms a ndarray Array1 to a vec
pub fn array1_to_vec<T>(array: Array1<T>) -> Vec<T>
where
    T: NdFloat,
{
    let mut result: Vec<T> = Vec::new();
    for i in 0..array.len() {
        result.push(array[i]);
    }
    result
}

/// Solves linear equation A*x = b by LU decomposition
pub fn matrix_solve<T>(a: &Array2<T>, b: &Array1<T>) -> Array1<T>
where
    T: NdFloat,
{
    //let x = a.solve(&b).expect("can't solve matrix");
    let matrix_dimension = a.shape()[0];
    // solve Ax = b for x, where A is a square matrix
    let (l, u, p) = lu_decomp(a);
    // first solve Ly = Pb

    let pivotized_b = p.dot(b);
    let mut y: Array1<T> = pivotized_b;
    for i in 1..matrix_dimension {
        let yi = y[i] - l.slice(s![i, 0..i]).dot(&y.slice(s![0..i]));
        y[i] = yi;
    }
    // then solve Ux = y
    let mut x: Array1<T> = y.clone();
    x[matrix_dimension - 1] /= u[[matrix_dimension - 1, matrix_dimension - 1]];
    for i in (0..matrix_dimension - 1).rev() {
        x[i] = (x[i]
            - u.slice(s![i, i + 1..matrix_dimension])
                .dot(&x.slice(s![i + 1..matrix_dimension])))
            / u[[i, i]];
    }
    x
}

/// Solves linear equation A*x = b where the partial pivoted LU decomposition of PA = LU is given
pub fn lu_matrix_solve<T>(l: &Array2<T>, u: &Array2<T>, p: &Array2<T>, b: &Array1<T>) -> Array1<T>
where
    T: NdFloat,
{
    let matrix_dimension = l.shape()[0];
    // first solve Ly = Pb
    let privatized_b = p.dot(b);
    let mut y: Array1<T> = privatized_b;
    for i in 1..matrix_dimension {
        let yi = y[i] - l.slice(s![i, 0..i]).dot(&y.slice(s![0..i]));
        y[i] = yi;
    }
    // then solve Ux = y
    let mut x: Array1<T> = y.clone();
    x[matrix_dimension - 1] /= u[[matrix_dimension - 1, matrix_dimension - 1]];
    for i in (0..matrix_dimension - 1).rev() {
        x[i] = (x[i]
            - u.slice(s![i, i + 1..matrix_dimension])
                .dot(&x.slice(s![i + 1..matrix_dimension])))
            / u[[i, i]];
    }
    x
}

/// Performs partial pivoted LU decomposition of A such that P A = LU
/// with L a lower triangular matrix and U an upper triangular matrix
/// A needs to be a square matrix
pub fn lu_decomp<T>(a: &Array2<T>) -> (Array2<T>, Array2<T>, Array2<T>)
where
    T: NdFloat,
{
    //let f = a.factorize_into().unwrap();
    let matrix_dimension = a.shape()[0];
    assert_eq!(
        matrix_dimension,
        a.shape()[1],
        "Tried LU decomposition with a non-square matrix."
    );
    let p = pivot(a);
    let pivotized_a = p.dot(a);

    let mut l: Array2<T> = Array::eye(matrix_dimension);
    let mut u: Array2<T> = Array::zeros((matrix_dimension, matrix_dimension));
    for idx_col in 0..matrix_dimension {
        // fill U
        for idx_row in 0..idx_col + 1 {
            u[[idx_row, idx_col]] = pivotized_a[[idx_row, idx_col]]
                - u.slice(s![0..idx_row, idx_col]).dot(&l.slice(s![idx_row, 0..idx_row]));
        }
        // fill L
        for idx_row in idx_col + 1..matrix_dimension {
            l[[idx_row, idx_col]] = (pivotized_a[[idx_row, idx_col]]
                - u.slice(s![0..idx_col, idx_col]).dot(&l.slice(s![idx_row, 0..idx_col])))
                / u[[idx_col, idx_col]];
        }
    }
    (l, u, p)
}

/// Pivot matrix A
fn pivot<T>(a: &Array2<T>) -> Array2<T>
where
    T: NdFloat,
{
    let matrix_dimension = a.shape()[0];
    let mut p: Array2<T> = Array::eye(matrix_dimension);
    for (i, column) in a.axis_iter(Axis(1)).enumerate() {
        // find idx of maximum value in column i
        let mut max_pos = i;
        for j in i..matrix_dimension {
            if column[max_pos].abs() < column[j].abs() {
                max_pos = j;
            }
        }
        // swap rows of P if necessary
        if max_pos != i {
            swap_rows(&mut p, i, max_pos);
        }
    }
    p
}

/// Swaps two rows of a matrix
fn swap_rows<T>(a: &mut Array2<T>, idx_row1: usize, idx_row2: usize)
where
    T: NdFloat,
{
    // to swap rows, get two ArrayViewMuts for the corresponding rows
    // and apply swap element wise using ndarray::Zip
    let (.., mut matrix_rest) = a.view_mut().split_at(Axis(0), idx_row1);
    let (row0, mut matrix_rest) = matrix_rest.view_mut().split_at(Axis(0), 1);
    let (_matrix_helper, mut matrix_rest) = matrix_rest.view_mut().split_at(Axis(0), idx_row2 - idx_row1 - 1);
    let (row1, ..) = matrix_rest.view_mut().split_at(Axis(0), 1);
    Zip::from(row0).and(row1).for_each(std::mem::swap);
}

#[cfg(feature = "plotly")]
fn graph_plot(id: usize, x: Vec<Q>, y: Vec<Q>) -> Box<plotly::traces::Scatter<Q, Q>> {
    let color = hex_num_to_rgb(COLORS[id]);
    let label = format!("{id}");

    Scatter::new(x, y)
        .mode(Mode::Markers)
        .name(&label)
        .text_array(vec![&label])
        .marker(
            Marker::new().color(Rgb::new(color[0], color[1], color[2])).size(2), //.line(Line::new().color(NamedColor::White).width(0.5)),
        )
}
#[cfg(feature = "plotly")]
pub fn plot_graph(show: bool, embedding: &Array2<Q>, colors_map: &[i32]) {
    let layout = Layout::new()
        .title(Title::new("Umap"))
        .x_axis(plotly::layout::Axis::new().show_grid(false).zero_line(false))
        .y_axis(plotly::layout::Axis::new().show_line(false));
    let mut plot = Plot::new();
    let mut data_plot: [(Vec<Q>, Vec<Q>); 10] = Default::default();

    embedding
        .axis_iter(Axis(0))
        .zip(colors_map.iter())
        .for_each(|(xy, &color_id)| {
            let color_id = color_id as usize;
            data_plot[color_id].0.push(xy[0]);
            data_plot[color_id].1.push(xy[1]);
        });

    data_plot
        .iter()
        .enumerate()
        .for_each(|(id, data)| plot.add_trace(graph_plot(id, data.0.clone(), data.1.clone())));

    plot.set_layout(layout);

    if show {
        plot.show();
    }

    //println!("{}", plot.to_inline_html(Some("digit_plot")));
}

pub fn hex_num_to_rgb(num: usize) -> [u8; 3] {
    let r = (num >> 16) as u8;
    let g = ((num >> 8) & 0x00FF) as u8;
    let b = (num & 0x0000_00FF) as u8;
    let ret: [u8; 3] = [r, g, b];
    ret
}

pub fn uniform(data: &mut [Q], a: Q, random: &mut impl Rng) {
    data.iter_mut().for_each(|v| *v = random.gen_range(-a..a));
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn solve_system_of_linear_equations() {
        let a: Array2<f64> = array![[1.0, 3.0, 5.0], [2.0, 4.0, 7.0], [1.0, 1.0, 0.0],];
        let b: Array1<f64> = array![1.0, 2.0, 3.0];
        let x = matrix_solve(&a, &b);
        assert_eq!(a.dot(&x), b);
    }
}
