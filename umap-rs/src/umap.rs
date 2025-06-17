use crate::dist::{DistanceType, Q};
use crate::{embedding, fuzzy, knn, optimize, optimize_original};
use ndarray::Array2;
use rand::rngs::SmallRng;
use rand::SeedableRng;
pub struct ProgressReporter {
    range: f64,
    raw_reporter: fn(f64),
}

impl ProgressReporter {
    pub fn new(raw_reporter: fn(f64)) -> ProgressReporter {
        ProgressReporter {
            range: 1.0,
            raw_reporter,
        }
    }
    pub fn new_scaled(reporter: &ProgressReporter, start: f64, end: f64) -> ProgressReporter {
        ProgressReporter {
            range: (end - start) * reporter.range,
            raw_reporter: reporter.raw_reporter,
        }
    }
    pub fn report(&self, progress: f64) {
        (self.raw_reporter)(self.range * progress)
    }
}

pub struct Umap {
    pub(crate) learning_rate: Q,
    pub(crate) local_connectivity: Q,
    pub(crate) min_dist: Q,
    pub(crate) negative_sample_rate: usize,
    pub(crate) repulsion_strength: Q,
    pub(crate) set_op_mix_ratio: Q,
    pub(crate) spread: Q,

    pub distance_type: DistanceType,

    pub n_neighbors: usize,
    embedded_dim: usize,

    custom_number_of_epochs: Option<usize>,
}

impl Umap {
    pub fn new(
        distance_type: Option<DistanceType>,
        dimensions: usize,
        min_dist: Q,
        spread: Q,
        n_neighbors: usize,
        custom_number_of_epochs: Option<usize>,
    ) -> Umap {
        if custom_number_of_epochs.filter(|&x| x == 0).is_some() {
            panic!("custom_number_of_epochs, if provided, must be greater than 0");
        }

        let distance_type = distance_type.unwrap_or_else(DistanceType::euclidean);

        Umap {
            learning_rate: 1.0,
            local_connectivity: 1.0,
            min_dist,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
            set_op_mix_ratio: 1.0,
            spread,
            distance_type,
            n_neighbors,
            embedded_dim: dimensions,
            custom_number_of_epochs,
        }
    }

    fn initialize_internal(
        &self,
        x: &Array2<Q>,
        seed: Option<u64>,
        num_threads: usize,
    ) -> (Array2<f64>, usize, Vec<usize>, Vec<usize>, Vec<f64>, SmallRng) {
        let mut random = SmallRng::seed_from_u64(seed.unwrap_or(0));

        let apply_fuzzy_combine = true;
        let embedded_dim = self.embedded_dim;
        let n_epochs = self.get_num_epochs(x.dim().0);

        let (knn_indices, knn_distances) = knn::nearest_neighbors(x, self.n_neighbors, self.distance_type, num_threads);

        // This part of the process very roughly accounts for 2/3 of the work (the remaining work is in the Step calls)
        let mut graph = fuzzy::fuzzy_simplicial_set(
            &knn_indices,
            &knn_distances,
            self.local_connectivity,
            self.set_op_mix_ratio,
            apply_fuzzy_combine,
            None,
            None,
        );

        let (head, tail, epochs_per_sample) =
            embedding::initialize_simplicial_set_embedding(&mut graph, n_epochs as Q, &mut random);

        let embedding = embedding::initialize_embedding(&graph, embedded_dim, &mut random);

        (embedding, n_epochs, head, tail, epochs_per_sample, random)
    }

    /// Initialize the original non-parallel UMAP algo. Used in CR 7.1/7.2 and SR
    pub fn initialize_fit(&self, x: &Array2<Q>, seed: Option<u64>, num_threads: usize) -> optimize_original::State {
        let (embedding, n_epochs, head, tail, epochs_per_sample, random) =
            self.initialize_internal(x, seed, num_threads);

        optimize_original::initialize_optimization(
            self,
            random,
            embedding,
            n_epochs,
            head,
            tail,
            epochs_per_sample,
            self.distance_type,
        )
    }

    /// Initialize the parallelized epoch-batched UMAP algo. Used in Xenium.
    pub fn initialize_fit_parallelized(&self, x: &Array2<Q>, seed: Option<u64>, num_threads: usize) -> optimize::State {
        let (embedding, n_epochs, head, tail, epochs_per_sample, _random) =
            self.initialize_internal(x, seed, num_threads);

        // Now, initialize the optimization steps
        optimize::initialize_optimization(
            self,
            seed.unwrap_or(0),
            embedding,
            n_epochs,
            head,
            tail,
            epochs_per_sample,
            self.distance_type,
        )
    }

    /// Gets the number of epochs for optimizing the projection
    pub(crate) fn get_num_epochs(&self, rows: usize) -> usize {
        if let Some(n) = self.custom_number_of_epochs {
            return n;
        }

        if rows <= 10_000 {
            500
        } else {
            200
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::test_data;
    use insta::assert_debug_snapshot;

    #[test]
    fn step_method_2d() {
        let umap = Umap::new(None, 2, 0.1, 1.0, 15, None);
        let data = test_data::get_test_data();
        let samples = data.len();
        let dim = data[0].len();
        let data0: Vec<Q> = data
            .iter()
            .flat_map(|v| v.iter().map(|&i| i as Q).collect::<Vec<Q>>())
            .collect();

        let data2 = Array2::from_shape_vec((samples, dim), data0).unwrap();

        let mut state = umap.initialize_fit(&data2, None, 1);
        state.optimize();

        assert_eq!(500, state.n_epochs);
    }

    #[test]
    fn powf_math() {
        // this test may fail to generate the same results on different platforms due to rounding differences
        // in the system implementation of powf -- this will result in different embeddings for
        // the same inputs. We can't switch to the libm impl because it's way too slow.
        // TODO to switch to fast but platform agnostic implementation of powf

        let mut res = vec![];

        for base in [0.0f64, 0.1, 0.2, 0.3, 0.99, 1.01, 1.1, 1.5, 2.0, 5.0, 10.0] {
            for pow in [
                -2.5f64, -2.0, -1.5, -1.0, -0.6, -0.5, -0.1, 0.1, 0.5, 0.75, 1.1, 2.0, 3.0,
            ] {
                let v = base.powf(pow);
                res.push((base, pow, v));
            }
        }

        assert_debug_snapshot!(res);
    }
}
