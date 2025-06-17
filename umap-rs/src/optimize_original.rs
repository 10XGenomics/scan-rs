//! Original UMAP Rust impl, that attempts to be as faithful to the original as possible
//! It cannot be easily parallelized, because position updates are applied immediately,
//! and influence the updates of other particles within the same iteration.
//! CR 7.1 and 7.2 use this implementation.
//!
use crate::curve_fit::Minimizer;
use crate::dist::{DistanceType, DistanceTypeImpl, Q};
use crate::func_1d::Func1D;
use crate::umap::Umap;
use ndarray::{array, Array1, Array2};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

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
    random: SmallRng,
}

impl Default for State {
    fn default() -> Self {
        State {
            current_epoch: 0,
            move_other: true,
            initial_alpha: 1.0,
            alpha: 1.0,
            gamma: 1.0,
            a: 1.5769434603113077,
            b: 0.8950608779109733,
            negative_sample_rate: 5,
            embedding: Array2::default((1, 1)),
            n_epochs: 500,
            distance_type: DistanceType::euclidean(),
            head: Vec::new(),
            tail: Vec::new(),
            epochs_per_sample: Vec::new(),
            epoch_of_next_sample: Vec::new(),
            epoch_of_next_negative_sample: Vec::new(),
            epochs_per_negative_sample: Vec::new(),
            random: SmallRng::seed_from_u64(0),
        }
    }
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

    pub fn optimize(&mut self) {
        while self.current_epoch < self.n_epochs {
            optimize_layout_step(self);
        }
    }

    pub fn step(&mut self) -> bool {
        if self.current_epoch < self.n_epochs {
            optimize_layout_step(self);
            true
        } else {
            false
        }
    }

    pub fn get_embedding(&self) -> &Array2<Q> {
        &self.embedding
    }
}

#[allow(clippy::too_many_arguments)]
pub fn initialize_optimization(
    umap: &Umap,
    random: SmallRng,
    embedding: Array2<Q>,
    n_epochs: usize,
    head: Vec<usize>,
    tail: Vec<usize>,
    epochs_per_sample: Vec<Q>,
    distance_type: DistanceType,
) -> State {
    let (a, b) = find_ab_params(umap.spread, umap.min_dist);

    let epochs_per_negative_sample = epochs_per_sample
        .iter()
        .map(|&e| e / (umap.negative_sample_rate as Q))
        .collect::<Vec<_>>();

    State {
        random,
        a,
        b,
        embedding,
        initial_alpha: umap.learning_rate,
        alpha: umap.learning_rate,
        gamma: umap.repulsion_strength,
        negative_sample_rate: umap.negative_sample_rate,
        move_other: true,
        n_epochs,
        distance_type,

        // Set the optimization routine state
        head,
        tail,
        epochs_per_sample: epochs_per_sample.clone(),
        epoch_of_next_sample: epochs_per_sample,
        epoch_of_next_negative_sample: epochs_per_negative_sample.clone(),
        epochs_per_negative_sample,

        current_epoch: 0,
    }
}

/// Squared Euclidean distance
#[inline]
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

pub fn step(state: &mut State) -> usize {
    let current_epoch = state.current_epoch;
    let number_of_epochs_to_complete = state.n_epochs;
    if current_epoch < number_of_epochs_to_complete {
        optimize_layout_step(state);
    }

    state.current_epoch
}

fn optimize_layout_step(state: &mut State) {
    for i in 0..state.epochs_per_sample.len() {
        match state.distance_type.0 {
            DistanceTypeImpl::Euclidean { .. } => iterate_euclidean(state, i),
            _ => iterate(state, i),
        }
    }

    state.alpha = state.initial_alpha * (1.0 - (state.current_epoch as Q) / (state.n_epochs as Q));
    state.current_epoch += 1;
}

#[inline]
pub fn iterate_euclidean(state: &mut State, i: usize) {
    if state.epoch_of_next_sample[i] > state.current_epoch as Q {
        return;
    }

    let j = state.head[i];
    let k = state.tail[i];

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

        state.embedding[[j, d]] += grad_d * alpha;
        if state.move_other {
            state.embedding[[k, d]] -= grad_d * alpha;
        }
    }

    state.epoch_of_next_sample[i] += state.epochs_per_sample[i];

    let n_neg_samples =
        (state.current_epoch as Q - state.epoch_of_next_negative_sample[i]) / state.epochs_per_negative_sample[i];

    for _ in 0..n_neg_samples.floor() as isize {
        let k = state.random.random_range(0..state.embedding.shape()[0]);

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

            state.embedding[[j, d]] += grad_d * alpha;
        }
    }

    state.epoch_of_next_negative_sample[i] += n_neg_samples * state.epochs_per_negative_sample[i];
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

#[inline]
fn iterate(state: &mut State, i: usize) {
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
        let k = state.random.random_range(0..state.embedding.shape()[0]);

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
