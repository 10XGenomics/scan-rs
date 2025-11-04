use crate::graph::{Edges, UnGraph};
use crate::Clustering;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use std::collections::HashMap;

/// Undirected graph with f32 node weights and f32 edge weights. Used to represent the network being clustered.
pub type Graph = UnGraph<f32, f32, u32>;

/// Container for the network graph.
pub struct Network {
    pub(crate) graph: Graph,
}

/// Iterator over pairs of (adjacent node id, edge_weight) for all neighbors of a chosen node.
pub struct NeighborAndWeightIter<'a> {
    edge_iter: Edges<'a, f32, u32>,
    home_node: usize,
}

impl Iterator for NeighborAndWeightIter<'_> {
    type Item = (usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        match self.edge_iter.next() {
            Some(edge_ref) => {
                assert!(edge_ref.source().index() as usize == self.home_node);
                let neighbor = edge_ref.target().index();
                Some((neighbor as usize, *edge_ref.weight() as f64))
            }
            None => None,
        }
    }
}

impl Network {
    /// Create a new empty network
    pub fn new() -> Network {
        Network {
            graph: Graph::with_capacity(0, 0),
        }
    }

    /// Create a new network from a graph
    pub fn new_from_graph(graph: Graph) -> Network {
        Network { graph }
    }

    /// Number of nodes in the graph
    pub fn nodes(&self) -> usize {
        self.graph.node_count() as usize
    }

    /// Get the node weight of `node`.
    pub fn weight(&self, node: usize) -> f64 {
        *self.graph.node_weight((node as u32).into()).unwrap() as f64
    }

    /// Iterate over edges connected to `node`
    fn edges(&'_ self, node: usize) -> Edges<'_, f32, u32> {
        self.graph.edges((node as u32).into())
    }

    /// Iterator over pairs of (adjacent node id, edge_weight) for all neighbors of `node`.
    pub fn neighbors(&'_ self, node: usize) -> NeighborAndWeightIter<'_> {
        NeighborAndWeightIter {
            edge_iter: self.edges(node),
            home_node: node,
        }
    }

    /// Get the total weight of all nodes in the graph
    pub fn get_total_node_weight(&self) -> f64 {
        let mut w = 0.0;
        for i in 0..self.nodes() {
            w += self.weight(i);
        }

        w
    }

    /// Get the total edge weight of all nodes in the graph
    pub fn get_total_edge_weight(&self) -> f64 {
        self.graph
            .edge_references()
            .fold(0.0, |acc, edge| acc + *edge.weight() as f64)
    }

    /// Get the total edge weight of all nodes in the graph
    pub fn get_total_edge_weight_par(&self) -> f64 {
        let mut partial_sums = vec![];

        // sum up the edge weights in parallelized over chunks, then sum the chunks to ensure
        // a deterministic result.  Just allow double counting of edge weights, and fix at the end
        // almost certainly faster than filtering on the fly.
        self.graph
            .edges
            .par_chunks(256)
            .map(|node_chunk| {
                node_chunk
                    .iter()
                    .map(|w| w.iter().fold(0.0, |acc, edge| acc + edge.weight as f64))
                    .sum::<f64>()
            })
            .collect_into_vec(&mut partial_sums);

        // divide edge sum by 2 to account for double-counting
        partial_sums.iter().sum::<f64>() / 2.0
    }

    /// Tabulate the total edge weight of each node into `result`
    pub fn get_total_edge_weight_per_node(&self, result: &mut Vec<f64>) {
        result.clear();

        for i in 0..self.nodes() {
            let mut w = 0.0;
            for e in self.edges(i) {
                w += *e.weight() as f64;
            }

            result.push(w);
        }
    }

    /// Creates a reduced (or aggregate) network based on a clustering.
    /// Each node in the reduced network corresponds to a cluster of nodes in
    /// the original network. The weight of a node in the reduced network equals
    /// the sum of the weights of the nodes in the corresponding cluster in the
    /// original network. The weight of an edge between two nodes in the reduced
    /// network equals the sum of the weights of the edges between the nodes in
    /// the two corresponding clusters in the original network.
    pub fn create_reduced_network(&self, clustering: &impl Clustering) -> Network {
        let mut cluster_g = Graph::with_capacity(clustering.num_clusters(), clustering.num_clusters() * 2);

        for i in 0..clustering.num_clusters() {
            let ni = cluster_g.add_node(0.0);
            assert_eq!(ni.index() as usize, i);
        }

        // add up node weights into cluster weights
        for n in self.graph.node_indices() {
            let cluster = clustering.get(n.index() as usize);

            let cluster_node_weight = cluster_g.node_weight_mut((cluster as u32).into()).unwrap();
            *cluster_node_weight += self.graph.node_weight(n).unwrap();
        }

        let mut edge_memo = HashMap::new();

        for e in self.graph.edge_references() {
            let c1 = clustering.get(e.source().index() as usize) as u32;
            let c2 = clustering.get(e.target().index() as usize) as u32;

            if c1 == c2 {
                continue;
            }

            let (min1, max1) = if c1 < c2 { (c1, c2) } else { (c2, c1) };

            *edge_memo.entry((min1, max1)).or_insert(0.0) += e.weight();
        }

        for (&(c1, c2), &weight) in edge_memo.iter() {
            cluster_g.add_edge(c1.into(), c2.into(), weight);
        }

        Network { graph: cluster_g }
    }

    /// Make a subnetwork for each cluster
    pub fn create_subnetworks_slow(&self, c: &impl Clustering) -> Vec<Network> {
        let mut subnetworks = Vec::new();

        for i in 0..c.num_clusters() {
            let sub = self.create_subnetwork_from_cluster_id(c, i);
            subnetworks.push(sub);
        }

        subnetworks
    }

    /// Make a subnetwork for each cluster
    pub fn create_subnetworks(&self, c: &impl Clustering) -> Vec<Network> {
        let mut graphs = Vec::new();
        let mut new_id_map = Vec::with_capacity(c.nodes());
        let mut counts = vec![0; c.num_clusters()];

        for _ in 0..c.num_clusters() {
            let g = Graph::with_capacity(0, 0);
            graphs.push(g);
        }

        for i in 0..self.nodes() {
            let c = c.get(i);

            let new_id: u32 = counts[c];
            new_id_map.push(new_id);
            counts[c] += 1;
            graphs[c].add_node(*self.graph.node_weight((i as u32).into()).unwrap());
        }

        for e in self.graph.edge_references() {
            let n1 = e.source().index() as usize;
            let c1 = c.get(n1);

            let n2 = e.target().index() as usize;
            let c2 = c.get(n2);

            if c1 == c2 {
                let new_id1 = new_id_map[n1];
                let new_id2 = new_id_map[n2];
                graphs[c1].add_edge(new_id1.into(), new_id2.into(), *e.weight());
            }
        }

        graphs.into_iter().map(|graph| Network { graph }).collect()
    }

    /// Make a sub-graph that only contains nodes assigned to the given `cluster` in the `node_labels` slice.
    fn create_subnetwork_from_cluster_id(&self, c: &impl Clustering, cluster: usize) -> Network {
        let subset = self.graph.filter_map(
            |idx, v| {
                if c.get(idx.index() as usize) == cluster {
                    Some(*v)
                } else {
                    None
                }
            },
            |e| Some(*e),
        );

        Network { graph: subset }
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}
