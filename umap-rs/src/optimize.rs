//! Updated UMAP impl, this impl is parallelized, but still deterministic.
//! Position updates are saved to the end of each optimization "epoch" and applied
//! atomically. This permits us to compute the updates in parallel without races.

use crate::curve_fit::Minimizer;
use crate::dist::{DistanceType, DistanceTypeImpl, Q};
use crate::func_1d::Func1D;
use crate::umap::Umap;
use ndarray::{array, Array1, Array2};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::collections::HashMap;

pub struct State {
    pub move_other: bool,
    pub initial_alpha: Q,

    pub gamma: Q,
    pub a: Q,
    pub b: Q,
    pub negative_sample_rate: usize,

    pub n_epochs: usize,
    pub current_epoch: usize,

    pub head: Vec<usize>,
    pub tail: Vec<usize>,
    pub epochs_per_sample: Vec<Q>,
    pub epochs_per_negative_sample: Vec<Q>,

    //mutable
    pub alpha: Q,
    pub embedding: Array2<Q>,
    pub epoch_of_next_sample: Vec<Q>,
    pub epoch_of_next_negative_sample: Vec<Q>,

    distance_type: DistanceType,
    seed: u64,

    state_updates: Vec<StateUpdate>,
}

impl State {
    pub fn num_samples(&self) -> usize {
        self.head.len()
    }

    pub fn num_points(&self) -> usize {
        self.embedding.shape()[0]
    }

    pub fn dim(&self) -> usize {
        self.embedding.shape()[1]
    }

    #[inline(never)]
    fn apply_updates(&mut self, updates: &[StateUpdate]) {
        // note: we sum to this intermediate array to slightly reduce rounding effect and preserve existing numerical
        // behavior
        let shape = updates[0].embedding.shape();
        let mut sum_embedding: Array2<Q> = Array2::zeros((shape[0], shape[1]));

        for update in updates {
            sum_embedding += &update.embedding;

            add_hashmap_to_vec(&mut self.epoch_of_next_sample, &update.epoch_of_next_sample);
            add_hashmap_to_vec(
                &mut self.epoch_of_next_negative_sample,
                &update.epoch_of_next_negative_sample,
            );
        }

        self.embedding += &sum_embedding;
    }

    pub fn optimize(&mut self) {
        self.optimize_multithreaded(1);
    }

    pub fn optimize_multithreaded(&mut self, num_threads: usize) {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        while self.current_epoch < self.n_epochs {
            optimize_layout_step(self, Some(&thread_pool));
        }
    }

    pub fn step(&mut self, thread_pool: &rayon::ThreadPool) -> bool {
        if self.current_epoch < self.n_epochs {
            optimize_layout_step(self, Some(thread_pool));
            true
        } else {
            false
        }
    }

    pub fn get_embedding(&self) -> &Array2<Q> {
        &self.embedding
    }
}

fn add_hashmap_to_vec(base: &mut [f64], update: &HashMap<usize, f64>) {
    for (idx, v) in update {
        base[*idx] += *v;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn initialize_optimization(
    umap: &Umap,
    seed: u64,
    embedding: Array2<Q>,
    n_epochs: usize,
    head: Vec<usize>,
    tail: Vec<usize>,
    epochs_per_sample: Vec<Q>,
    distance_type: DistanceType,
) -> State {
    let (a, b) = find_ab_params(umap.spread, umap.min_dist);

    assert_eq!(head.len(), tail.len());
    assert_eq!(head.len(), epochs_per_sample.len());

    let epoch_of_next_sample = epochs_per_sample.clone();

    let epochs_per_negative_sample = epochs_per_sample
        .iter()
        .map(|&e| e / (umap.negative_sample_rate as Q))
        .collect::<Vec<_>>();

    let epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    let updates = StateUpdate {
        embedding: Array2::zeros((embedding.shape()[0], embedding.shape()[1])),
        epoch_of_next_sample: HashMap::new(),
        epoch_of_next_negative_sample: HashMap::new(),
    };

    State {
        current_epoch: 0,
        move_other: true,
        initial_alpha: umap.learning_rate,
        alpha: umap.learning_rate,
        gamma: umap.repulsion_strength,
        a,
        b,
        negative_sample_rate: umap.negative_sample_rate,
        embedding,
        n_epochs,
        distance_type,
        head,
        tail,
        epochs_per_sample,
        epoch_of_next_sample,
        epoch_of_next_negative_sample,
        epochs_per_negative_sample,
        // this set of StateUpdates set the chunking that will be used
        // for parallelization
        state_updates: vec![updates; 16],
        seed,
    }
}

/// Squared Euclidean distance
#[inline(never)]
fn euclidean_sq(embeddings: &Array2<Q>, j: usize, k: usize) -> Q {
    let x = embeddings.row(j);
    let y = embeddings.row(k);
    x.iter().zip(y).map(|(&x, &y)| x - y).fold(0.0, |acc, s| acc + s * s)
}

fn curve(p: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    x.map(|&val| 1.0 / (1.0 + p[0] * val.powf(2.0 * p[1])))
}

fn find_ab_params(spread: Q, min_dist: Q) -> (Q, Q) {
    let p = array![2.0, 1.0];
    let x: Array1<f64> = Array1::range(0.0, 3.0 * spread, spread / 100.0);

    let y: Array1<f64> = x.map(|&x| {
        if x < min_dist {
            1.0
        } else {
            (-(x - min_dist) / spread).exp()
        }
    });
    let model = Func1D::new(&p, &x, curve);
    let sy = Array1::from(vec![1.0; x.len()]);
    let vary_parameter = array![true, true];
    let lambda = 1.0;
    let mut minimizer = Minimizer::init(&model, &y, &sy, &vary_parameter, lambda);
    minimizer.minimize();
    minimizer.report();
    (
        minimizer.minimizer_parameters[0] as Q,
        minimizer.minimizer_parameters[1] as Q,
    )
}

#[derive(Clone)]
struct StateUpdate {
    embedding: Array2<Q>,
    epoch_of_next_sample: HashMap<usize, f64>,
    epoch_of_next_negative_sample: HashMap<usize, f64>,
}

impl StateUpdate {
    fn clear(&mut self) {
        self.embedding.fill(0.0);
        self.epoch_of_next_sample.clear();
        self.epoch_of_next_negative_sample.clear();
    }
}

fn optimize_layout_step(state: &mut State, thread_pool: Option<&rayon::ThreadPool>) {
    match state.distance_type.0 {
        DistanceTypeImpl::Euclidean { .. } => {
            if let Some(tp) = thread_pool {
                // go to multi-threaded method
                optimize_layout_euclidean_threaded(state, tp)
            } else {
                unreachable!("don't run the single-threaded version -- it will give different results than the thread-pool version due different FP order-of-operations");

                /*
                // single-threaded version
                let mut updates = StateUpdate {
                    embedding: Array2::zeros((state.embedding.shape()[0], state.embedding.shape()[1])),
                    epoch_of_next_sample: vec![0f64; state.num_samples()],
                    epoch_of_next_negative_sample: vec![0f64; state.num_samples()],
                };

                for i in 0..state.num_samples() {
                    iterate_euclidean(state, i, &mut updates);
                }
                state.apply_update(&updates);
                */
            }
        }
        _ => {
            // TODO: Non-Euclidean distance version has not been parallelized yet
            for i in 0..state.num_samples() {
                iterate(state, i);
            }
        }
    }

    state.alpha = state.initial_alpha * (1.0 - (state.current_epoch as Q) / (state.n_epochs as Q));
    state.current_epoch += 1;
}

// multi-threaded version of the Euclidean UMAP update.
// this parallelization method is chosen to
// avoid non-determinism arising from dynamic work-stealing.
// each epoch + sample gets a deterministic seed, which should avoid
// random seed differences related to threading
// causing different random seeds and/or different FP order-of-operations.
// this method will have a different FP order-of-operations with different
// thread count.
#[inline(never)]
fn optimize_layout_euclidean_threaded(state: &mut State, thread_pool: &rayon::ThreadPool) {
    // We reuse the StateUpdates for each iteration to avoid having to reallocate them each time.
    // pull out the StateUpdate buffers from storage
    let mut updates = vec![];
    std::mem::swap(&mut updates, &mut state.state_updates);

    // divide the samples into a fixed number of chunks and then parallelize over those chunks
    // this prevents small results changes as a function of the thread count.
    let chunk_size = (state.num_samples() / updates.len()) + 1;
    let mut start = 0;

    let mut ranges = vec![];
    loop {
        if start > state.num_samples() {
            break;
        };
        let end = std::cmp::min(start + chunk_size, state.num_samples());
        ranges.push(start..end);
        start += chunk_size;
    }

    // Each chunk fills up one StateUpdate which get put into finished_update
    let mut finished_updates = vec![];

    thread_pool.install(|| {
        ranges
            .into_par_iter()
            .zip(updates)
            .map(|(range, mut my_chunk_updates)| {
                for i in range {
                    iterate_euclidean(state, i, &mut my_chunk_updates);
                }
                my_chunk_updates
            })
            .collect_into_vec(&mut finished_updates);
    });

    // apply the updates
    state.apply_updates(&finished_updates);

    // reset the updates
    for update in finished_updates.iter_mut() {
        update.clear();
    }

    // store the StateUpdates back ino State for reuse
    std::mem::swap(&mut finished_updates, &mut state.state_updates);
}

#[inline(never)]
fn iterate_euclidean(state: &State, i: usize, update: &mut StateUpdate) {
    // each epoch + i gets a deterministic seed
    let seed = state.seed ^ ((state.current_epoch as u64) << 32 | (i as u64));
    let mut random = SmallRng::seed_from_u64(seed);

    if state.epoch_of_next_sample[i] > state.current_epoch as Q {
        return;
    }

    let j = state.head[i];
    let k = state.tail[i];
    assert!(j != k);

    let (a, b, gamma, alpha) = (state.a, state.b, state.gamma, state.alpha);
    let embedded_dim = state.embedding.shape()[1];

    let dist_sq = euclidean_sq(&state.embedding, j, k);
    let grad_coeff = if dist_sq > 0.0 {
        (-2.0 * a * b * dist_sq.powf(b - 1.0)) / (1.0 + a * dist_sq.powf(b))
    } else {
        0.0
    };

    for d in 0..embedded_dim {
        let current = state.embedding[[j, d]];
        let other = state.embedding[[k, d]];
        let grad_d = (grad_coeff * (current - other)).clamp(-4.0, 4.0);

        update.embedding[[j, d]] += grad_d * alpha;
        if state.move_other {
            update.embedding[[k, d]] -= grad_d * alpha;
        }
    }

    update.epoch_of_next_sample.insert(i, state.epochs_per_sample[i]);

    let n_neg_samples =
        (state.current_epoch as Q - state.epoch_of_next_negative_sample[i]) / state.epochs_per_negative_sample[i];

    for _ in 0..n_neg_samples.floor() as isize {
        let k = random.random_range(0..state.num_points());

        if j == k {
            continue;
        }

        let dist_sq = euclidean_sq(&state.embedding, j, k);
        let grad_coeff = if dist_sq > 0.0 {
            (2.0 * gamma * b) / ((1e-3 + dist_sq) * (1.0 + a * dist_sq.powf(b)))
        } else {
            0.0
        };

        for d in 0..embedded_dim {
            let current = state.embedding[[j, d]];
            let other = state.embedding[[k, d]];
            let grad_d = if grad_coeff > 0.0 {
                (grad_coeff * (current - other)).clamp(-4.0, 4.0)
            } else {
                4.0
            };

            update.embedding[[j, d]] += grad_d * alpha;
        }
    }

    update
        .epoch_of_next_negative_sample
        .insert(i, n_neg_samples * state.epochs_per_negative_sample[i]);
}

fn output_metric(
    embeddings: &Array2<Q>,
    j: usize,
    k: usize,
    distance_fn: crate::dist::DistanceGradFn,
) -> (Q, Array1<Q>) {
    let x = embeddings.row(j).to_slice().unwrap();
    let y = embeddings.row(k).to_slice().unwrap();
    (distance_fn)(x, y)
}

#[inline(never)]
fn iterate(state: &mut State, i: usize) {
    // each epoch + i gets a deterministic seed
    let seed = state.seed ^ ((state.current_epoch as u64) << 32 | (i as u64));
    let mut random = SmallRng::seed_from_u64(seed);

    if state.epoch_of_next_sample[i] > state.current_epoch as Q {
        return;
    }

    let j = state.head[i];
    let k = state.tail[i];

    let (a, b, gamma, alpha) = (state.a, state.b, state.gamma, state.alpha);
    let embedded_dim = state.embedding.shape()[1];

    let distance_grad_fn = match state.distance_type.0 {
        DistanceTypeImpl::Other { grad, .. } => grad,
        _ => panic!("unreachable"),
    };

    let (dist_output, grad_dist_output) = output_metric(&state.embedding, j, k, distance_grad_fn);
    let (_, rev_grad_dist_output) = output_metric(&state.embedding, k, j, distance_grad_fn);

    let w_l = if dist_output > 0.0 {
        1.0 / (1.0 + a * dist_output.powf(2. * b))
    } else {
        1.0
    };

    let grad_coeff = 2.0 * b * (w_l - 1.0) / (dist_output + 1e-6);

    for d in 0..embedded_dim {
        let grad_d = (grad_coeff * grad_dist_output[d]).clamp(-4.0, 4.0);

        state.embedding[[j, d]] += grad_d * alpha;
        if state.move_other {
            let grad_d = (grad_coeff * rev_grad_dist_output[d]).clamp(-4.0, 4.0);
            state.embedding[[k, d]] += grad_d * alpha;
        }
    }

    state.epoch_of_next_sample[i] += state.epochs_per_sample[i];

    let n_neg_samples =
        (state.current_epoch as Q - state.epoch_of_next_negative_sample[i]) / state.epochs_per_negative_sample[i];

    for _ in 0..n_neg_samples.floor() as i32 {
        let k = random.random_range(0..state.embedding.shape()[0]);

        let (dist_output, grad_dist_output) = output_metric(&state.embedding, j, k, distance_grad_fn);

        if dist_output <= 0.0 && j == k {
            continue;
        }

        let w_l = if dist_output > 0.0 {
            1.0 / (1.0 + a * dist_output.powf(2. * b))
        } else {
            1.0
        };

        let grad_coeff = gamma * 2.0 * b * w_l / (dist_output + 1e-6);

        for d in 0..embedded_dim {
            let grad_d = (grad_coeff * grad_dist_output[d]).clamp(-4.0, 4.0);
            state.embedding[[j, d]] += grad_d * alpha;
        }
    }

    state.epoch_of_next_negative_sample[i] += n_neg_samples * state.epochs_per_negative_sample[i];
}
