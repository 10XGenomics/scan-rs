#![allow(non_snake_case, clippy::upper_case_acronyms)]
#![deny(warnings)]

#[macro_use]
extern crate smart_default;

use ndarray::Array2;
use std::os::raw::c_int;
mod bindings;
pub use bindings::TSNE as TSNEImpl;

struct TSNE(*mut TSNEImpl);

impl Default for TSNE {
    fn default() -> Self {
        TSNE(std::ptr::null_mut())
    }
}

impl Drop for TSNE {
    fn drop(&mut self) {
        unsafe { bindings::free_tsne(self.0) }
    }
}

// np.random.RandomState(0).randint(2**31-1)
const RANDOM_STATE: c_int = 209_652_396;
const STOP_LYING_ITER: usize = 250;
const MOM_SWITCH_ITER: usize = 250;

#[derive(SmartDefault)]
pub struct BarnesHutTSNE {
    #[default = 2]
    pub n_dims: u32,
    #[default = 50.]
    pub perplexity: f64,
    #[default = 0.5]
    pub theta: f64,
    #[default(None)]
    pub seed: Option<u32>,
    #[default = 1000]
    pub max_iter: usize,
    #[default(None)]
    pub stop_lying_iter: Option<usize>,
    #[default(None)]
    pub mom_switch_iter: Option<usize>,
    #[default(Array2::default((0, 0)))]
    Y: Array2<f64>,
    #[default(TSNE::default())]
    tsne: TSNE,
}

impl BarnesHutTSNE {
    pub fn init(&mut self, X: &mut Array2<f64>) {
        assert!(X.is_standard_layout());

        let N = X.shape()[0];
        let D = X.shape()[1];

        self.Y = Array2::zeros((N, self.n_dims as usize));

        unsafe {
            self.tsne.0 = bindings::init_tsne(
                X.as_mut_ptr(),
                N as c_int,
                D as c_int,
                self.Y.as_mut_ptr(),
                self.n_dims as c_int,
                self.perplexity,
                self.theta,
                self.seed.map(|x| x as c_int).unwrap_or(RANDOM_STATE),
                false,
                std::ptr::null_mut(),
                false,
                self.max_iter as c_int,
                self.stop_lying_iter.unwrap_or(STOP_LYING_ITER) as c_int,
                self.mom_switch_iter.unwrap_or(MOM_SWITCH_ITER) as c_int,
            );
        }
    }

    pub fn run_n(&mut self, n: usize) -> bool {
        unsafe { bindings::step_tsne_by(self.tsne.0, n as c_int) }
    }

    pub fn result(self) -> Array2<f64> {
        self.Y
    }
}
