use itertools::Itertools;
pub use kodama::Method as LinkageMethod;
use kodama::{linkage, Dendrogram, Float as KodamaFloat, Step};
use ndarray::{Array2, ArrayView1};
use num_traits::Float;
use petgraph::unionfind::UnionFind;
use std::collections::HashMap;
use std::ops::Mul;

#[derive(Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
}

impl DistanceMetric {
    pub fn apply<T: Float + Mul>(self, x1: &ArrayView1<T>, x2: &ArrayView1<T>) -> T {
        match self {
            DistanceMetric::Euclidean => {
                let mut dx = x2 - x1;
                dx.map_inplace(|v| {
                    *v = (*v) * (*v);
                });
                dx.sum().sqrt()
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum ClusterDirection {
    // Treat each row as an observation
    Rows,
    // Treat each column as an observation
    Columns,
}

impl ClusterDirection {
    pub fn n<F: Float>(self, array: &Array2<F>) -> usize {
        match self {
            ClusterDirection::Rows => array.nrows(),
            ClusterDirection::Columns => array.ncols(),
        }
    }
    pub fn get<F: Float>(self, array: &Array2<F>, index: usize) -> ArrayView1<F> {
        match self {
            ClusterDirection::Rows => array.row(index),
            ClusterDirection::Columns => array.column(index),
        }
    }
}

#[derive(Clone, Copy)]
pub enum LeafOrdering {
    /// When merging two clusters, always keep the one with lower index on the left
    /// and the one with higher index on the right
    Naive,
    /// See <https://f1000research.com/articles/3-177/v1>
    ModularSmallest,
}

#[derive(Debug, Default, Clone, Copy)]
struct Boundary {
    left: Option<usize>,
    right: Option<usize>,
}

struct SimpleOrdering {
    // Left and right most leaves in each cluster
    boundaries: Vec<Boundary>,
    // Left and right leaves of each leaf node
    neighbor_leaves: Vec<Boundary>,
}

impl SimpleOrdering {
    fn new(n_obs: usize) -> Self {
        // n_obs observations and n_obs-1 clustering steps
        // Left and right most leaves in each cluster
        let mut boundaries = vec![Boundary::default(); 2 * n_obs - 1];

        // For singleton clusters, left and right are the same
        for (i, boundary) in boundaries.iter_mut().enumerate().take(n_obs) {
            boundary.left = Some(i);
            boundary.right = Some(i);
        }

        SimpleOrdering {
            boundaries,
            neighbor_leaves: vec![Boundary::default(); n_obs],
        }
    }
    fn observe(&mut self, merged_cluster: usize, left_cluster: usize, right_cluster: usize) {
        self.boundaries[merged_cluster] = Boundary {
            left: self.boundaries[left_cluster].left,
            right: self.boundaries[right_cluster].right,
        };

        self.neighbor_leaves[self.boundaries[left_cluster].right.unwrap()].right = self.boundaries[right_cluster].left;
        self.neighbor_leaves[self.boundaries[right_cluster].left.unwrap()].left = self.boundaries[left_cluster].right;
    }
    fn ordered_leaves(self) -> Vec<usize> {
        let (start_node, _) = self.neighbor_leaves.iter().find_position(|b| b.left.is_none()).unwrap();

        let mut leaves = vec![start_node];
        loop {
            let next = self.neighbor_leaves[*leaves.last().unwrap()].right;
            match next {
                Some(leaf) => leaves.push(leaf),
                None => break,
            }
        }

        assert_eq!(leaves.len(), self.neighbor_leaves.len());

        leaves
    }
}

/// Relabel vector of cluster labelling so that cluster names
/// start with 1 and are consecutive integers.
/// For example, [5, 3, 5, 5, 10, 15, 10, 15] maps to
/// [1, 2, 1, 1, 3, 4, 3, 4]
fn relabel_vector(input: &[usize]) -> Vec<usize> {
    let mapping: HashMap<_, _> = input.iter().unique().enumerate().map(|(i, v)| (v, i + 1)).collect();
    input.iter().map(|x| mapping[x]).collect_vec()
}

pub struct HierarchicalCluster<F: Float> {
    dendrogram: Dendrogram<F>,
}

impl<F: Float + KodamaFloat> HierarchicalCluster<F> {
    pub fn new(array: &Array2<F>, metric: DistanceMetric, method: LinkageMethod, direction: ClusterDirection) -> Self {
        let n = direction.n(array);
        if n < 2 {
            panic!("Need at least two elements to do hierarchical clustering");
        }
        let mut condensed_dissimilarity = vec![];

        for i in 0..n {
            let x_i = direction.get(array, i);
            for j in i + 1..n {
                let x_j = direction.get(array, j);
                condensed_dissimilarity.push(metric.apply(&x_i, &x_j));
            }
        }

        let dendrogram = linkage(&mut condensed_dissimilarity, n, method);
        assert_eq!(dendrogram.len(), n - 1);
        HierarchicalCluster { dendrogram }
    }

    fn naive_leaf_ordering(&self) -> Vec<usize> {
        let mut ordering = SimpleOrdering::new(self.dendrogram.observations());
        for (merged_cluster, Step { cluster1, cluster2, .. }) in self.steps_with_cluster_num() {
            let (left_cluster, right_cluster) = if cluster1 < cluster2 {
                (*cluster1, *cluster2)
            } else {
                (*cluster2, *cluster1)
            };
            ordering.observe(merged_cluster, left_cluster, right_cluster);
        }
        ordering.ordered_leaves()
    }

    fn steps_with_cluster_num(&self) -> impl Iterator<Item = (usize, &Step<F>)> {
        let n_obs = self.dendrogram.observations();
        self.dendrogram
            .steps()
            .iter()
            .enumerate()
            .map(move |(i, s)| (i + n_obs, s))
    }

    fn total_nodes(&self) -> usize {
        self.dendrogram.observations() + self.dendrogram.len()
    }

    fn modular_smallest_leaf_ordering(&self) -> Vec<usize> {
        let mut min_dissimilarity: Vec<F> = vec![num_traits::Float::max_value(); self.total_nodes()];
        for (merged_cluster, step) in self.steps_with_cluster_num() {
            min_dissimilarity[merged_cluster] = min_dissimilarity[step.cluster1]
                .min(min_dissimilarity[step.cluster2])
                .min(step.dissimilarity);
        }
        let mut ordering = SimpleOrdering::new(self.dendrogram.observations());
        for (merged_cluster, Step { cluster1, cluster2, .. }) in self.steps_with_cluster_num() {
            let (left_cluster, right_cluster) = if min_dissimilarity[*cluster1] <= min_dissimilarity[*cluster2] {
                (*cluster1, *cluster2)
            } else {
                (*cluster2, *cluster1)
            };
            ordering.observe(merged_cluster, left_cluster, right_cluster);
        }
        ordering.ordered_leaves()
    }
    /// Order the leaves according to the ordering algorithm specified
    ///
    /// Returns
    /// - x : A vector with length equal to the number of nodes.
    ///       x[i] = j implies that node "j" is present at index "i"
    ///       when reading the leaves from left to right
    pub fn leaves(&self, ordering: LeafOrdering) -> Vec<usize> {
        match ordering {
            LeafOrdering::Naive => self.naive_leaf_ordering(),
            LeafOrdering::ModularSmallest => self.modular_smallest_leaf_ordering(),
        }
    }

    /// Get hierarchical clustering labels by merging all links
    /// where the distance is below a threshold
    pub fn merge_clusters_below_distance_threshold(&self, distance_threshold: F) -> Vec<usize> {
        let num_points = self.dendrogram.observations();
        let mut union_find_clsts = UnionFind::new(num_points);
        let mut new_clusters_to_old_map = HashMap::new();
        for (index, stp) in self
            .dendrogram
            .steps()
            .iter()
            .enumerate()
            .filter(|(_, sp)| sp.dissimilarity < distance_threshold)
        {
            let new_cluster1 = *new_clusters_to_old_map.get(&stp.cluster1).unwrap_or(&stp.cluster1);
            let new_cluster2 = *new_clusters_to_old_map.get(&stp.cluster2).unwrap_or(&stp.cluster2);
            union_find_clsts.union(new_cluster1, new_cluster2);
            new_clusters_to_old_map.insert(num_points + index, union_find_clsts.find(new_cluster1));
        }
        relabel_vector(&union_find_clsts.into_labeling())
    }

    /// Get hierarchical clusters when we list cluster
    /// down to num_clusters clusters
    pub fn fcluster(&self, num_clusters: &usize) -> Vec<usize> {
        let num_points = self.dendrogram.observations();
        let distance_threshold = if *num_clusters <= 1 {
            return vec![1; num_points];
        } else {
            self.dendrogram.steps()[num_points.saturating_sub(*num_clusters)].dissimilarity
        };
        self.merge_clusters_below_distance_threshold(distance_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_relabel() {
        assert_eq!(
            vec![1, 2, 1, 1, 3, 4, 3, 4],
            relabel_vector(&[5, 3, 5, 5, 10, 15, 10, 15])
        );
    }

    #[test]
    fn test_fcluster() {
        // # Python code to reconstruct this test
        // import numpy as np
        // from scipy.cluster import hierarchy
        // arr = np.array([
        //             [4, 5, 10, 4, 3, 11, 14, 6, 10, 12],
        //             [21, 19, 24, 17, 16, 25, 24, 22, 21, 21],
        //             [13, 10, 42, 7, 1, 17, 14, 20, 11, 9]
        //         ])
        //
        // def relabel_cluster(clusters):
        //     count = 1
        //     cluster_to_base_clusters = {}
        //     for ind, x in enumerate(clusters):
        //         if x not in cluster_to_base_clusters:
        //             cluster_to_base_clusters[x] = count
        //             count += 1
        //     return list(map(lambda x: cluster_to_base_clusters[x], clusters))
        //
        // link_topics = hierarchy.linkage(arr.T, "ward", "euclidean")
        // print(link_topics)
        // >> array([[ 8.        ,  9.        ,  2.82842712,  2.        ],
        //           [ 0.        ,  1.        ,  3.74165739,  2.        ],
        //           [ 5.        ,  6.        ,  4.35889894,  2.        ],
        //           [ 3.        ,  4.        ,  6.164414  ,  2.        ],
        //           [10.        , 12.        ,  9.46044396,  4.        ],
        //           [ 7.        , 11.        , 10.23067284,  3.        ],
        //           [14.        , 15.        , 13.40486763,  7.        ],
        //           [13.        , 16.        , 21.33407737,  9.        ],
        //           [ 2.        , 17.        , 41.50421665, 10.        ]])
        //
        // dist_thresholds = list(
        //         reversed(
        //             [0.5 * (link_topics[i][2] + link_topics[i + 1][2]) for i in range(len(link_topics) - 1)]
        //         )
        //     ) + [0.0]
        // dist_thresholds = [link_topics[-1][2]+1] + dist_thresholds
        // for dist in dist_thresholds:
        //     print(relabel_cluster(hierarchy.fcluster(link_topics, dist, criterion="distance")))
        // >>  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        //     [1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
        //     [1, 1, 2, 3, 3, 1, 1, 1, 1, 1]
        //     [1, 1, 2, 3, 3, 4, 4, 1, 4, 4]
        //     [1, 1, 2, 3, 3, 4, 4, 5, 4, 4]
        //     [1, 1, 2, 3, 3, 4, 4, 5, 6, 6]
        //     [1, 1, 2, 3, 4, 5, 5, 6, 7, 7]
        //     [1, 1, 2, 3, 4, 5, 6, 7, 8, 8]
        //     [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
        //     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        // 10 points living in R^3 to cluster on
        let arr = array![
            [4, 5, 10, 4, 3, 11, 14, 6, 10, 12],
            [21, 19, 24, 17, 16, 25, 24, 22, 21, 21],
            [13, 10, 42, 7, 1, 17, 14, 20, 11, 9],
        ]
        .mapv(|x| x as f32);

        let cluster = HierarchicalCluster::new(
            &arr,
            DistanceMetric::Euclidean,
            LinkageMethod::Ward,
            ClusterDirection::Columns,
        );
        assert_eq!(vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cluster.fcluster(&0));
        assert_eq!(vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cluster.fcluster(&1));
        assert_eq!(vec![1, 1, 2, 1, 1, 1, 1, 1, 1, 1], cluster.fcluster(&2));
        assert_eq!(vec![1, 1, 2, 3, 3, 1, 1, 1, 1, 1], cluster.fcluster(&3));
        assert_eq!(vec![1, 1, 2, 3, 3, 4, 4, 1, 4, 4], cluster.fcluster(&4));
        assert_eq!(vec![1, 1, 2, 3, 3, 4, 4, 5, 4, 4], cluster.fcluster(&5));
        assert_eq!(vec![1, 1, 2, 3, 3, 4, 4, 5, 6, 6], cluster.fcluster(&6));
        assert_eq!(vec![1, 1, 2, 3, 4, 5, 5, 6, 7, 7], cluster.fcluster(&7));
        assert_eq!(vec![1, 1, 2, 3, 4, 5, 6, 7, 8, 8], cluster.fcluster(&8));
        assert_eq!(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 9], cluster.fcluster(&9));
        assert_eq!(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cluster.fcluster(&10));
        assert_eq!(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cluster.fcluster(&11));
    }

    #[test]
    fn test_leaf_order() {
        // From https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
        let arr = array![
            [4, 5, 10, 4, 3, 11, 14, 6, 10, 12],      //  x - coordinates
            [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]  // y - coordinates
        ]
        .mapv(|x| x as f32);

        assert_eq!(arr.shape(), &[2, 10]);

        assert_eq!(ClusterDirection::Rows.n(&arr), 2);
        assert_eq!(ClusterDirection::Rows.get(&arr, 0).len(), 10);
        assert_eq!(ClusterDirection::Columns.n(&arr), 10);
        assert_eq!(ClusterDirection::Columns.get(&arr, 0).len(), 2);

        let cluster = HierarchicalCluster::new(
            &arr,
            DistanceMetric::Euclidean,
            LinkageMethod::Ward,
            ClusterDirection::Columns,
        );

        println!("{:#?}", cluster.dendrogram);

        assert_eq!(cluster.leaves(LeafOrdering::Naive), vec![8, 9, 6, 2, 5, 3, 4, 7, 0, 1]);
        assert_eq!(
            cluster.leaves(LeafOrdering::ModularSmallest),
            vec![2, 5, 6, 8, 9, 3, 4, 0, 1, 7]
        );
    }

    #[test]
    #[should_panic]
    fn test_empty_array() {
        let _ = HierarchicalCluster::<f32>::new(
            &array![[], []],
            DistanceMetric::Euclidean,
            LinkageMethod::Ward,
            ClusterDirection::Columns,
        )
        .leaves(LeafOrdering::ModularSmallest);
    }

    #[test]
    #[should_panic]
    fn test_single_element_array() {
        let _ = HierarchicalCluster::<f32>::new(
            &array![[1.0], [1.0]],
            DistanceMetric::Euclidean,
            LinkageMethod::Ward,
            ClusterDirection::Columns,
        )
        .leaves(LeafOrdering::ModularSmallest);
    }
}
