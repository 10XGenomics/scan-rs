/// Trait representing a clustering over a set of items
/// Each item is assigned a single integer label
pub trait Clustering: std::fmt::Debug {
    /// Initialize a fresh clustering with each node in it's own cluster
    fn init_different_clusters(num_nodes: usize) -> Self;

    /// Initialize a fresh clustering with all nodes in a single cluser
    fn init_same_cluster(num_nodes: usize) -> Self;

    /// Initialize the clustering with a known set of labels
    fn new_from_labels(labels: &[usize]) -> Self;

    /// Tabulate the number of nodes in each cluster
    fn nodes_per_cluster(&self) -> Vec<Vec<usize>>;

    /// Get the label of node `i`
    fn get(&self, i: usize) -> usize;

    /// Set the label of node 'i'. Note this must update the number of clusters
    fn set(&mut self, i: usize, cluster: usize);

    /// Total number of nodes
    fn nodes(&self) -> usize;

    /// Number of distinct clusters
    fn num_clusters(&self) -> usize;

    /// Reassign node labels, removing unused labels
    fn remove_empty_clusters(&mut self);

    /// Take clustering of the cluster labels, and reassign label to reflect the higher-order clustering
    fn merge_clusters<C: Clustering>(&mut self, cluster_clusterings: &C) {
        for i in 0..self.nodes() {
            self.set(i, cluster_clusterings.get(self.get(i)))
        }

        self.remove_empty_clusters();
    }

    /// Set all cluster labels to 0.
    fn clear(&mut self) {
        for i in 0..self.nodes() {
            self.set(i, 0);
        }

        self.remove_empty_clusters();
    }
}

#[derive(Debug, Clone, Default)]
/// A basic Vec-backed implementation of `Clustering`
pub struct SimpleClustering {
    labels: Vec<usize>,
    num_clusters: usize,
}

impl Clustering for SimpleClustering {
    fn init_different_clusters(num_nodes: usize) -> Self {
        let mut labels = Vec::with_capacity(num_nodes);

        for i in 0..num_nodes {
            labels.push(i);
        }

        SimpleClustering {
            labels,
            num_clusters: num_nodes,
        }
    }

    fn init_same_cluster(num_nodes: usize) -> Self {
        let labels = vec![0; num_nodes];

        SimpleClustering {
            labels,
            num_clusters: usize::from(num_nodes > 0),
        }
    }

    fn new_from_labels(input_labels: &[usize]) -> Self {
        let mut labels = Vec::with_capacity(input_labels.len());

        let mut max_cluster = 0;

        for l in input_labels.iter() {
            max_cluster = std::cmp::max(*l, max_cluster);
            labels.push(*l);
        }

        let mut r = SimpleClustering {
            labels,
            num_clusters: max_cluster + 1,
        };

        r.remove_empty_clusters();
        r
    }

    fn nodes_per_cluster(&self) -> Vec<Vec<usize>> {
        let mut cluster_lists = vec![Vec::new(); self.num_clusters()];

        for (node, label) in self.labels.iter().enumerate() {
            cluster_lists[*label].push(node)
        }

        cluster_lists
    }

    fn get(&self, node: usize) -> usize {
        self.labels[node]
    }

    fn set(&mut self, node: usize, label: usize) {
        self.labels[node] = label;
        if label >= self.num_clusters {
            self.num_clusters = label + 1;
        }
    }

    fn nodes(&self) -> usize {
        self.labels.len()
    }

    fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    fn remove_empty_clusters(&mut self) {
        let mut counts = vec![0; self.num_clusters()];

        for &l in self.labels.iter() {
            counts[l] += 1;
        }

        let mut new_labels = Vec::with_capacity(self.num_clusters());

        let mut new_label = 0;
        for cluster_count in counts {
            if cluster_count == 0 {
                new_labels.push(std::usize::MAX);
            } else {
                new_labels.push(new_label);
                new_label += 1;
            }
        }

        for i in 0..self.labels.len() {
            let old_label = self.labels[i];
            let new_label = new_labels[old_label];
            assert!(new_label != std::usize::MAX);
            self.labels[i] = new_label;
        }

        self.num_clusters = new_label;
    }
}
/*
#[derive(Debug)]
pub struct AdjClustering {
    num_clusters: usize,
    labels: Vec<usize>,
}

impl Clustering for AdjClustering {
    fn new(num_nodes: usize) -> AdjClustering {
        AdjClustering {
            labels: vec![0; num_nodes],
            num_clusters: if num_nodes > 0 { 1 } else { 0 }
        }
    }

    fn new_from_labels(labels: &[usize]) -> AdjClustering {
        let mut num_clusters = 0;
        for l in labels {
            num_clusters = std::cmp::max(num_clusters, l + 1);
        }

        AdjClustering {
            labels: Vec::from(labels),
            num_clusters
        }
    }

    fn get(&self, node: usize) -> usize {
        self.labels[node]
    }

    fn set(&mut self, node: usize, cluster: usize) {
        self.labels[node] = cluster;
        if cluster+1 > self.num_clusters {
            self.num_clusters += 1;
        }
    }

    fn nodes_per_cluster(&self) -> Vec<Vec<usize>> {
        let mut node_lists = Vec::new();
        for _ in 0 .. self.num_clusters {
            node_lists.push(Vec::new());
        }

        for (idx, label) in self.labels.iter().enumerate() {
            node_lists[*label].push(idx)
        }

        node_lists
    }


    fn nodes(&self) -> usize {
        self.labels.len()
    }

    fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    fn remove_empty_clusters(&mut self) {
        println!("self: {:#?}", self);

        let mut counts = vec![0; self.num_clusters()];
        for c in self.labels.iter() {
            counts[*c] += 1;
        }

        let mut new_labels = vec![0; self.num_clusters()];

        let mut new_label_counter = 0;
        for old_label in 0 .. self.num_clusters() {
            new_labels[old_label] = new_label_counter;
            if counts[old_label] > 0 {
                new_label_counter += 1;
            }
        }

        self.num_clusters = new_label_counter;

        for i in 0 .. self.labels.len() {
            let new = new_labels[self.labels[i]];
            self.labels[i] = new;
        }
    }
}
*/

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_from_labels() {
        let labels = vec![1, 2, 3, 4, 5];
        let mut c = SimpleClustering::new_from_labels(&labels);
        assert_eq!(c.num_clusters(), 5);

        c.remove_empty_clusters();
        assert_eq!(c.num_clusters(), 5);
        assert_eq!(&c.labels, &vec![0, 1, 2, 3, 4])
    }

    #[test]
    fn test_operations() {
        let mut c = SimpleClustering::init_different_clusters(10);
        assert_eq!(c.num_clusters(), 10);
        println!("c: {c:?}");

        c.set(8, 0);
        c.set(7, 0);
        println!("c: {c:?}");

        c.remove_empty_clusters();
        println!("c: {c:?}");

        assert_eq!(c.num_clusters(), 8);
        assert_eq!(c.get(9), 7);
    }
}
