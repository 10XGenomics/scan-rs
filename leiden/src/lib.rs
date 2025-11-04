//! Leiden community detection algorithm
#![deny(missing_docs)]
#![deny(warnings)]

/// Data structure for storing a clustering of nodes
pub mod clustering;

/// Data structure for storing a weighted, undirected graph (aka network)
pub mod network;

/// Leiden clustering algorithm
pub mod leiden;

/// Louvain clustering algorithm
pub mod louvain;
/// Parallelized louvain clustering algorithm
pub mod louvain_parallel;

/// Clustering objective functions
pub mod objective;

mod fast_local_moving;
mod graph;
mod local_merging;
mod parallel_local_moving;
mod standard_local_moving;

#[cfg(test)]
mod test;

pub use clustering::{Clustering, SimpleClustering};
pub use network::{Graph, Network};

trait ZeroVec {
    fn zero(&mut self);
    fn zero_len(&mut self, len: usize);
}

impl<T: Default> ZeroVec for Vec<T> {
    fn zero(&mut self) {
        for i in self.iter_mut() {
            *i = T::default();
        }
    }

    fn zero_len(&mut self, len: usize) {
        self.zero();
        self.resize_with(len, T::default)
    }
}
