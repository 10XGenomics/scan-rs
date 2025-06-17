use crate::fast_local_moving::FastLocalMoving;
use crate::local_merging::LocalMerging;
use crate::{Clustering, Network, ZeroVec};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Perform the Leiden clustering algorithm
pub struct Leiden {
    resolution: f64,
    randomness: f64,

    rng: SmallRng,

    local_moving: FastLocalMoving,
    num_nodes_per_cluster_reduced_network: Vec<usize>,
}

impl Leiden {
    /// Initialize the Leiden algorithm with the given resolution and randomness parameters.
    /// An optional random seed can be supplied, otherwise a seed of 0 will be used.
    pub fn new(resolution: f64, randomness: f64, seed: Option<usize>) -> Leiden {
        let seed = seed.unwrap_or_default() as u64;

        Leiden {
            resolution,
            randomness,
            rng: SmallRng::seed_from_u64(seed),
            local_moving: FastLocalMoving::new(resolution),
            num_nodes_per_cluster_reduced_network: Vec::new(),
        }
    }

    /// Iterate the Leiden algorithm one step. Returns true if cluster labels were updated, otherwise returns false.
    pub fn iterate<C: Clustering>(&mut self, n: &Network, c: &mut C) -> bool {
        // Update the clustering by moving individual nodes between clusters.
        let mut update = self.local_moving.iterate(n, c, &mut self.rng);

        if c.num_clusters() == n.nodes() {
            return update;
        }

        let mut local_merging = LocalMerging::new(self.resolution, self.randomness);

        let subnetworks = n.create_subnetworks(c);

        let nodes_per_cluster = c.nodes_per_cluster();

        // clear clustering
        c.clear();

        self.num_nodes_per_cluster_reduced_network.zero_len(subnetworks.len());
        let mut cluster_counter = 0;

        for i in 0..subnetworks.len() {
            let sub_clustering = local_merging.run(&subnetworks[i], &mut self.rng);

            for j in 0..subnetworks[i].nodes() {
                c.set(nodes_per_cluster[i][j], cluster_counter + sub_clustering.get(j))
            }

            cluster_counter += sub_clustering.num_clusters();
            self.num_nodes_per_cluster_reduced_network[i] = sub_clustering.num_clusters()
        }
        c.remove_empty_clusters();

        // Create an aggregate network based on the refined clustering of
        // the non-aggregate network.
        let reduced_n = n.create_reduced_network(c);

        // Create an initial clustering for the aggregate network based on the
        // non-refined clustering of the non-aggregate network.
        let mut clusters_reduced_network = vec![0; c.num_clusters()];

        let mut i = 0;
        for (j, num_nodes) in self.num_nodes_per_cluster_reduced_network.iter().enumerate() {
            for cluster in clusters_reduced_network.iter_mut().skip(i).take(*num_nodes) {
                *cluster = j;
            }
            i += num_nodes;
        }

        let mut clustering_reduced_network = C::new_from_labels(&clusters_reduced_network);

        // Recursively apply the algorithm to the aggregate network,
        // starting from the initial clustering created for this network.
        update |= self.iterate(&reduced_n, &mut clustering_reduced_network);

        // Update the clustering of the non-aggregate network so that it
        // coincides with the final clustering obtained for the aggregate
        // network.
        c.merge_clusters(&clustering_reduced_network);

        update
    }
}
