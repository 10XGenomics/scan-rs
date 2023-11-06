//! Large test of UMAP using mocked up single-cell data

#[cfg(test)]
#[allow(clippy::module_inception)]
pub mod rand_test {
    use std::{
        io::Error,
        path::{Path, PathBuf},
    };

    use ndarray::{Array1, Array2};
    use rand_distr::{Distribution, Gamma, Uniform};
    use rand_pcg::Pcg64Mcg;

    use ndarray_rand::RandomExt;
    use rand::SeedableRng;
    use rand_distr::Normal;
    use serde::{de::DeserializeOwned, Deserialize, Serialize};

    use crate::{dist::DistanceType, umap::Umap};
    const N_NEIGHBORS: usize = 30;
    const MIN_DIST: f64 = 0.3;
    const N_COMPONENTS: usize = 2;

    type Q = crate::utils::Q;

    fn seeded_rng(seed: u64) -> Pcg64Mcg {
        Pcg64Mcg::seed_from_u64(seed)
    }

    fn simulated_cells(cells: usize, num_clusters: usize, sigma: f64, seed: u64) -> (Array2<Q>, Array1<usize>) {
        let mut rng = seeded_rng(seed);
        let n = 20;

        let mut clusters = Vec::new();
        for _ in 0..num_clusters {
            let r = Gamma::new(0.4, 2.0).unwrap();
            let a = Array1::<f64>::random_using(n, r, &mut rng);
            clusters.push(a);
        }

        let mut mat = Array2::zeros((cells, n));
        let mut cluster_labels = Array1::zeros(cells);

        let c = Uniform::new(0, clusters.len());
        for i in 0..cells {
            let cluster_id = c.sample(&mut rng);
            cluster_labels[i] = cluster_id;
            let row = &clusters[cluster_id];

            for j in 0..n {
                let dist = Normal::new(row[j], sigma).unwrap();
                let v = dist.sample(&mut rng);

                mat[(i, j)] = v;
            }
        }

        (mat, cluster_labels)
    }

    fn run_umap(projection: &Array2<Q>, parallel: bool) -> Array2<Q> {
        let metric = DistanceType::pearson();

        let spread = 1.0;
        let umap = Umap::new(Some(metric), N_COMPONENTS, MIN_DIST, spread, N_NEIGHBORS, None);

        let embedding = if parallel {
            let mut state = umap.initialize_fit_parallelized(projection, None);
            state.optimize_multithreaded(2);
            state.optimize();
            state.get_embedding().clone()
        } else {
            let mut state = umap.initialize_fit_parallelized(projection, None);
            state.optimize_multithreaded(2);
            state.get_embedding().clone()
        };

        embedding
    }

    #[test]
    fn run_large_umap() {
        // simulated PCA projection
        let (simulated_proj, _labels) = simulated_cells(20000, 20, 1., 0);
        let _embedding = run_umap(&simulated_proj, true);
    }

    /// Check for exact matching of results. Due to our use of the system libm (via f64::powf) the results
    /// will change across machines with different libm implementations - e.g. different versions of Linux and Mac
    /// Therefore this test is disabled for now.
    #[ignore]
    #[test]
    fn umap_validation_test_parallel() {
        let prefix = "test_results_parallel";
        let cmp_prefix = "expected_results_parallel";

        std::fs::create_dir(prefix).unwrap();

        let res = [
            run_umap_test(cmp_prefix, prefix, "small-distinct", 500, 3, 0.75, 0, true),
            run_umap_test(cmp_prefix, prefix, "small-blurry", 500, 5, 1.1, 1, true),
            run_umap_test(cmp_prefix, prefix, "med-distinct", 1500, 5, 0.75, 2, true),
            run_umap_test(cmp_prefix, prefix, "med-blurry", 1500, 7, 1.1, 3, true),
            run_umap_test(cmp_prefix, prefix, "large-distinct", 1500, 5, 0.75, 4, true),
            run_umap_test(cmp_prefix, prefix, "large-blurry", 1500, 7, 1.1, 5, true),
        ];

        let messages: Vec<_> = res.into_iter().flatten().collect();

        // remove this line to save outputs, in case they require updating.
        let _ = std::fs::remove_dir_all(prefix);

        if !messages.is_empty() {
            panic!("{messages:?}");
        }
    }

    /// Check for exact matching of results. Due to our use of the system libm (via f64::powf) the results
    /// will change across machines with different libm implementations - e.g. different versions of Linux and Mac
    /// Therefore this test is disabled for now.
    #[ignore]
    #[test]
    fn umap_validation_test_orig() {
        let prefix = "test_results_orig";
        let cmp_prefix = "expected_results_orig";

        std::fs::create_dir(prefix).unwrap();

        let res = [
            run_umap_test(cmp_prefix, prefix, "small-distinct", 500, 3, 0.75, 0, false),
            run_umap_test(cmp_prefix, prefix, "small-blurry", 500, 5, 1.1, 1, false),
            run_umap_test(cmp_prefix, prefix, "med-distinct", 1500, 5, 0.75, 2, false),
            run_umap_test(cmp_prefix, prefix, "med-blurry", 1500, 7, 1.1, 3, false),
            run_umap_test(cmp_prefix, prefix, "large-distinct", 1500, 5, 0.75, 4, false),
            run_umap_test(cmp_prefix, prefix, "large-blurry", 1500, 7, 1.1, 5, false),
        ];

        let messages: Vec<_> = res.into_iter().flatten().collect();

        // remove this line to save outputs, in case they require updating.
        let _ = std::fs::remove_dir_all(prefix);

        if !messages.is_empty() {
            panic!("{messages:?}");
        }
    }

    #[derive(Serialize, Deserialize)]
    struct EmbeddingPoint {
        prefix: String,
        name: String,
        label: usize,
        x: Q,
        y: Q,
    }

    #[allow(clippy::too_many_arguments)]
    fn run_umap_test(
        cmp_prefix: &str,
        prefix: &str,
        name: &str,
        cells: usize,
        num_clusters: usize,
        sigma: f64,
        seed: u64,
        parallel: bool,
    ) -> Option<String> {
        // simulated PCA projection
        println!("{name}: simulate");
        let (simulated_proj, labels) = simulated_cells(cells, num_clusters, sigma, seed);

        println!("{name}: embed");
        let embedding = run_umap(&simulated_proj, parallel);

        println!("{name}: write");
        write_umap(prefix, name, &embedding, &labels).unwrap();

        println!("{name}: read");
        let (expected_embedding, _) = read_umap(cmp_prefix, name);

        println!("{name}: compare");
        compare_embeddings(name, &embedding, &expected_embedding)
    }

    fn compare_embeddings(name: &str, a: &Array2<Q>, b: &Array2<Q>) -> Option<String> {
        if !a.abs_diff_eq(b, 0.01) {
            Some(format!("abs diff violated in {name}"))
        } else if !a.relative_eq(b, 0.01, 0.01) {
            Some(format!("rel diff violated in {name}"))
        } else {
            None
        }
    }

    fn write_umap(prefix: &str, name: &str, embedding: &Array2<Q>, labels: &Array1<usize>) -> Result<(), Error> {
        let mut path = PathBuf::from(prefix);
        path.push(format!("{name}.csv"));

        let file = std::fs::File::create(path).unwrap();
        let mut writer = csv::WriterBuilder::new().has_headers(true).from_writer(file);

        for cell in 0..embedding.shape()[0] {
            let pt = EmbeddingPoint {
                prefix: prefix.to_string(),
                name: name.to_string(),
                label: labels[cell],
                x: embedding[(cell, 0)],
                y: embedding[(cell, 1)],
            };

            writer.serialize(pt).unwrap();
        }

        Ok(())
    }

    fn read_umap(prefix: &str, name: &str) -> (Array2<Q>, Array1<usize>) {
        let mut path = PathBuf::from(prefix);
        path.push(format!("{name}.csv"));

        let pts: Vec<EmbeddingPoint> = load_tabular(path, true, b',').unwrap();

        let mut embedding = Array2::zeros((pts.len(), 2));
        let mut labels = Array1::zeros(pts.len());

        for (i, p) in pts.iter().enumerate() {
            embedding[(i, 0)] = p.x;
            embedding[(i, 1)] = p.y;
            labels[i] = p.label;
        }

        (embedding, labels)
    }

    /// load tabular data of type T from a possibly gzipped tabular file
    pub fn load_tabular<T: DeserializeOwned>(
        f: impl AsRef<Path>,
        headers: bool,
        delimiter: u8,
    ) -> Result<Vec<T>, Error> {
        let reader = std::fs::File::open(f).unwrap();

        let mut csv = csv::ReaderBuilder::default()
            .delimiter(delimiter)
            .has_headers(headers)
            .flexible(true)
            .from_reader(reader);

        let mut items = Vec::new();

        for f in csv.deserialize() {
            items.push(f?);
        }

        Ok(items)
    }
}
