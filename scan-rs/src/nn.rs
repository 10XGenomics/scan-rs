use ball_tree::{BallTree, Point};
use log::info;
use ndarray::parallel::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use ndarray::{Array2, ArrayView2, Axis};
use num_traits::bounds::Bounded;
use num_traits::cast::FromPrimitive;
use num_traits::identities::Zero;

#[derive(PartialEq)]
struct Pt(Vec<f64>);

impl Point for Pt {
    fn distance(&self, other: &Self) -> f64 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|p| (p.1 - p.0).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn move_towards(&self, other: &Self, d: f64) -> Self {
        let total_dist = self.distance(other);
        let frac = if total_dist == 0.0 { 0.0 } else { d / total_dist };

        Pt(self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(&s, &o)| s + frac * (o - s))
            .collect())
    }
}

/// Compute the `k` nearest neighbors of each row in `v`, using Euclidean distance. Each row represents a n-dimensional
/// vector where n is the number of columns in `v`.
pub fn knn<T>(v: &ArrayView2<f64>, k: usize) -> Array2<T>
where
    T: Bounded + Clone + Copy + FromPrimitive + Send + Sync + Zero,
{
    let mut points = Vec::new();
    let mut values = Vec::new();
    let (cells, _) = v.dim();

    info!("constructing ball tree of {} points", cells);
    for i in 0..cells {
        let pt = Pt(Vec::from(v.row(i).as_slice().unwrap()));
        points.push(pt);
        values.push(i);
    }
    let ball_tree = BallTree::new(points, values);

    info!("querying points for {} neighbors", k);
    let mut output = Array2::from_elem((cells, k), T::max_value());
    output.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each_init(
        || ball_tree.query(),
        |query, (cell, mut output)| {
            let mut ind = 0;
            let pt = Pt(Vec::from(v.row(cell).as_slice().unwrap()));
            for (_, _, &v) in query.nn(&pt).take(k + 1) {
                if v != cell && ind < k {
                    output[ind] = T::from_usize(v).expect("unrepresentable!!");
                    ind += 1;
                }
            }
        },
    );
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;
    use ndarray_rand::RandomExt;
    use ordered_float::NotNan;
    use rand::SeedableRng;
    use rand_distr::Normal;
    use rand_pcg::Pcg64Mcg;

    fn distance(v1: &[f64], other: &[f64]) -> NotNan<f64> {
        let mut d = 0.0;
        for i in 0..v1.len() {
            d += (other[i] - v1[i]).powi(2);
        }

        NotNan::new(d.sqrt()).unwrap()
    }

    // Basic n^2 knn algorithm, for testing purposes
    fn exhaustive_knn(v: &ArrayView2<f64>, k: usize) -> Array2<usize> {
        let cells = v.shape()[0];
        let mut nns = Vec::new();
        assert!(k < cells);

        let mut output = Array2::zeros((cells, k));

        for cell in 0..cells {
            nns.clear();

            let my_row = v.row(cell);
            let my_point = my_row.as_slice().unwrap();

            for other_cell in 0..cells {
                if cell == other_cell {
                    continue;
                }

                let other_row = v.row(other_cell);
                let other_point = other_row.as_slice().unwrap();

                let d = distance(my_point, other_point);
                nns.push((d, other_cell));
            }

            nns.sort();

            for i in 0..k {
                output[(cell, i)] = nns[i].1
            }
        }

        output
    }

    fn validate_knn(v: &ArrayView2<f64>) {
        let full_knn = exhaustive_knn(v, std::cmp::min(v.shape()[0] - 1, 50));

        for k in &[1, 5, 10, 25, 50] {
            if k >= &v.shape()[0] {
                continue;
            }

            let fast_knn = knn::<usize>(v, *k);
            let slow_knn = full_knn.slice(s![.., 0..*k]).to_owned();

            assert_eq!(fast_knn, slow_knn);
        }
    }

    #[test]
    fn test_knn() {
        let mut rng = Pcg64Mcg::seed_from_u64(0);

        for ncells in &[3, 5, 50, 100] {
            for d in &[1, 2, 3, 5, 10, 20, 50] {
                let dist = Normal::new(0.0f64, 1.0f64).unwrap();
                let v = Array2::<f64>::random_using((*ncells, *d), dist, &mut rng);
                validate_knn(&v.view());
            }
        }
    }

    #[test]
    fn test_symmetry() {
        // A bunch of equally distant points, with one outlier
        let mut v = Array2::<f64>::eye(5);
        v[(0, 4)] = 3.0f64;

        let knn = knn::<usize>(&v.view(), 4);

        // There are some degeneracies in the distances,
        // but the ball tree is still deterministc
        let correct = ndarray::arr2(&[[4, 2, 1, 3], [4, 2, 3, 0], [4, 1, 3, 0], [4, 1, 2, 0], [2, 1, 3, 0]]);

        println!("{knn:?}");
        assert_eq!(knn, correct);
    }
}
