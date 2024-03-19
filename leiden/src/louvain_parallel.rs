use crate::parallel_local_moving::ParallelLocalMoving;
use crate::{Clustering, Graph, Network};
use std::collections::HashSet;

/// Perform the Louvain clustering algorithm
pub struct ParallelLouvain {
    local_moving: ParallelLocalMoving,
}

impl ParallelLouvain {
    /// Initialize the Leiden algorithm with the given resolution and randomness parameters.
    pub fn new(resolution: f64) -> ParallelLouvain {
        ParallelLouvain {
            local_moving: ParallelLocalMoving::new(resolution),
        }
    }

    /// Iterate the Louvain algorithm a single level
    pub fn iterate_one_level<C: Clustering + Clone + Send + Sync + Default>(&mut self, n: &Network, c: &mut C) -> bool {
        self.local_moving.iterate(n, c)
    }

    /// Build a Louvain-compatible network from a list of adjacencies
    pub fn build_network<I: Iterator<Item = (u32, u32)>>(n_nodes: usize, n_edges: usize, adjacency: I) -> Network {
        let mut graph = Graph::with_capacity(n_nodes, n_edges);
        let mut node_indices = Vec::with_capacity(n_nodes);
        for _ in 0..n_nodes {
            node_indices.push(graph.add_node(1.0));
        }
        let mut seen = vec![HashSet::<u32>::new(); n_nodes];
        let mut node_weights = vec![0.0; n_nodes];
        for (i, j) in adjacency {
            let (i, j) = if i < j { (i, j) } else { (j, i) };
            let i_ = i as usize;
            let j_ = j as usize;
            if seen[i_].insert(j) {
                graph.add_edge(i.into(), j.into(), 1.0);
                // weights are just degree here
                node_weights[i_] += 1.0;
                node_weights[j_] += 1.0;
            }
        }
        // reweight nodes based on # of edges
        for &i in &node_indices {
            *graph.node_weight_mut(i).unwrap() = node_weights[i.index() as usize];
        }
        Network::new_from_graph(graph)
    }
}
