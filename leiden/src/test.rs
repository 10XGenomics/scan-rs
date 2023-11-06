use crate::leiden::Leiden;
use crate::louvain::Louvain;
use crate::louvain_parallel::ParallelLouvain;
use crate::objective::{cpm, par_cpm};
use crate::{Clustering, Graph, Network, SimpleClustering};
use flate2::read::GzDecoder;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::time::{Duration, Instant};

/// Generate a random test graph using the method described in section 4B of the Leiden paper.
fn gen_sample_network(
    rng: &mut impl Rng,
    num_clusters: usize,
    nodes_per_cluster: usize,
    mean_degree: f64,
    mu: f64,
) -> (Graph, SimpleClustering) {
    assert!(num_clusters > 1);
    assert!(nodes_per_cluster > 1);

    let total_nodes = num_clusters * nodes_per_cluster;
    let total_edges = (total_nodes as f64 * mean_degree).ceil() as usize;

    // which cluster is each node assigned to?
    let mut cluster = Vec::with_capacity(total_nodes);

    for c in 0..num_clusters {
        for _ in 0..nodes_per_cluster {
            cluster.push(c);
        }
    }
    let true_clusters = SimpleClustering::new_from_labels(&cluster);

    let mut g = Graph::with_capacity(total_nodes, total_edges);

    for _ in 0..total_nodes {
        g.add_node(1.0);
    }

    for _ in 0..total_edges {
        let in_cluster = rng.gen_bool(1.0 - mu);

        // Choose Node 1
        let n1 = rng.gen_range(0..total_nodes);
        let c1 = true_clusters.get(n1);

        // Choose Node 2 -- optimize the case when the node2 should be in the same cluster as node1
        let mut n2 = if in_cluster {
            rng.gen_range(c1 * nodes_per_cluster..(c1 + 1) * nodes_per_cluster)
        } else {
            rng.gen_range(0..total_nodes)
        };

        let mut c2 = true_clusters.get(n2);

        // Iterate until node2 is of the kind we want
        loop {
            if n1 != n2 && in_cluster && c1 == c2 {
                break;
            }

            if n1 != n2 && !in_cluster && c1 != c2 {
                break;
            }

            n2 = rng.gen_range(0..total_nodes);
            c2 = cluster[n2];
        }

        g.add_edge((n1 as u32).into(), (n2 as u32).into(), 1.0);
        g.add_edge((n2 as u32).into(), (n1 as u32).into(), 1.0);
    }

    (g, true_clusters)
}

const DEFAULT_RESOLUTION: f64 = 1.0;
const DEFAULT_RANDOMNESS: f64 = 1e-2;
const DEFAULT_EPSILON: f64 = 1e-6;

#[test]
fn run_leiden() {
    let mut rng = ChaCha20Rng::seed_from_u64(0);

    let num_clusters = 100000 / 50;
    let nodes_per_cluster = 50;

    let (g, true_clusters) = gen_sample_network(&mut rng, num_clusters, nodes_per_cluster, 10.0, 0.4);
    let n = Network { graph: g };

    println!("best cpm: {}", cpm(DEFAULT_RESOLUTION, &n, &true_clusters));

    let mut c = SimpleClustering::init_different_clusters(n.nodes());
    let mut l = Leiden::new(DEFAULT_RESOLUTION, DEFAULT_RANDOMNESS, None);

    let score = cpm(DEFAULT_RESOLUTION, &n, &c);
    println!("initial cpm: {score}");

    for i in 0..10 {
        let update = l.iterate(&n, &mut c);

        let score = cpm(DEFAULT_RESOLUTION, &n, &c);
        println!("iter: {i}, cpm: {score}");

        if !update {
            break;
        }
    }
}

fn relabel_by_size(labels: &mut [i16]) -> Vec<(i16, usize)> {
    let max_label = labels.iter().max().unwrap();
    let mut hist = (0..(max_label + 1)).map(|i| (i, 0usize)).collect::<Vec<_>>();
    labels.iter().for_each(|&x| hist[x as usize].1 += 1);
    hist.sort_by(|(_, x), (_, y)| y.cmp(x));
    let map = hist
        .iter()
        .enumerate()
        .map(|(i, &j)| (j.0, i as i16))
        .collect::<HashMap<_, _>>();
    for x in labels.iter_mut() {
        *x = map[x];
    }
    for kv in hist.iter_mut() {
        kv.0 = map[&kv.0];
    }
    hist
}

fn rand_index(x: &[i16], y: &[i16]) -> f64 {
    assert!(x.len() == y.len(), "x.len({}) != y.len({})", x.len(), y.len());
    let n = x.len();
    let mut num = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let xi_eq_xj = x[i] == x[j];
            let yi_eq_yj = y[i] == y[j];
            if xi_eq_xj == yi_eq_yj {
                num += 1;
            }
        }
    }
    let den = n * (n - 1) / 2;
    (num as f64 / den as f64).min(1.0)
}

#[test]
#[ignore] // needs test fixture
fn run_louvain() -> std::io::Result<()> {
    let (n_nodes, adjacency) = {
        let file = BufReader::new(GzDecoder::new(File::open("testdata/louvain/adjacency.txt.gz")?));
        let mut nodes = HashSet::new();
        let mut adjacency = Vec::new();
        for line in file.lines() {
            let line = line?;
            let mut iter = line.split_ascii_whitespace();
            let a = iter.next().unwrap().parse::<u32>().unwrap();
            let b = iter.next().unwrap().parse::<u32>().unwrap();
            nodes.insert(a);
            nodes.insert(b);
            adjacency.push((a, b));
        }
        (nodes.len(), adjacency)
    };
    let network = Louvain::build_network(n_nodes, adjacency.len(), adjacency.into_iter());
    println!(
        "nodes: {:?}\nedges: {:?}",
        network.graph.node_count(),
        network.graph.edge_count()
    );
    let resolution = crate::louvain::DEFAULT_RESOLUTION;
    let mut clustering: SimpleClustering = Clustering::init_different_clusters(n_nodes);
    let mut louvain = Louvain::new(resolution, Some(0xBADC0FFEE0DDF000));
    let mut score = cpm(resolution, &network, &clustering);
    println!("initial cpm: {score:.8}");
    for iter in 1.. {
        let updated = louvain.iterate_one_level(&network, &mut clustering);
        let new_score = cpm(resolution, &network, &clustering);
        if !updated || new_score - score <= DEFAULT_EPSILON {
            score = new_score;
            break;
        }
        score = new_score;
        println!("iteration {iter} cpm: {score:.8}");
    }
    println!("final cpm: {score:.8}");

    let mut res: Vec<i16> = Vec::with_capacity(n_nodes);
    let mut max_label = 0;
    for i in 0..n_nodes {
        let label = clustering.get(i) as i16;
        if max_label < label {
            max_label = label;
        }
        res.push(label);
    }
    // sort from greatest to least
    let hist = relabel_by_size(&mut res);
    println!("test cluster sizes {:?}", &hist);
    let expected = {
        let file = BufReader::new(GzDecoder::new(File::open("testdata/louvain/expected.txt.gz")?));
        let mut assn = vec![];
        for line in file.lines() {
            let line = line?;
            let mut iter = line.split_ascii_whitespace().skip(1);
            assn.push(iter.next().unwrap().parse::<i16>().unwrap());
        }
        let hist = relabel_by_size(&mut assn);
        println!("expected cluster sizes {:?}", &hist);
        assn
    };
    let ri = rand_index(&res, &expected);
    println!("rand index: {ri}");
    assert!(1.0 - ri < 1e-2);
    Ok(())
}

#[test]
#[ignore] // needs test fixture
fn run_louvain_parallel() -> std::io::Result<()> {
    // Use to scale up dataset size. Incompatible with rand index check at the end.
    let n_repeats = 0;

    let (n_nodes, adjacency) = {
        let data: Vec<u8> = BufReader::new(GzDecoder::new(File::open("testdata/louvain/adjacency.txt.gz")?))
            .bytes()
            .collect::<Result<_, _>>()?;

        let mut nodes = HashSet::new();
        let mut adjacency = Vec::new();
        let mut max_orig_node = 0;
        for line in BufReader::new(data.as_slice()).lines() {
            let line = line?;
            let mut iter = line.split_ascii_whitespace();
            let a = iter.next().unwrap().parse::<u32>().unwrap();
            let b = iter.next().unwrap().parse::<u32>().unwrap();
            nodes.insert(a);
            nodes.insert(b);
            adjacency.push((a, b));
            max_orig_node = std::cmp::max(max_orig_node, std::cmp::max(a, b));
        }

        // Duplicate it a few times
        for replicate in 1..n_repeats {
            for line in BufReader::new(data.as_slice()).lines() {
                let line = line?;
                let mut iter = line.split_ascii_whitespace();
                let a = iter.next().unwrap().parse::<u32>().unwrap() + replicate * max_orig_node;
                let b = iter.next().unwrap().parse::<u32>().unwrap() + replicate * max_orig_node;
                nodes.insert(a);
                nodes.insert(b);
                adjacency.push((a, b));
            }
        }
        (nodes.len(), adjacency)
    };
    let network = Louvain::build_network(n_nodes, adjacency.len(), adjacency.into_iter());
    println!(
        "nodes: {:?}\nedges: {:?}",
        network.graph.node_count(),
        network.graph.edge_count()
    );
    let resolution = crate::louvain::DEFAULT_RESOLUTION;
    let mut clustering: SimpleClustering = Clustering::init_different_clusters(n_nodes);
    let mut louvain = ParallelLouvain::new(resolution);
    let mut score = par_cpm(resolution, &network, &clustering);
    println!("initial cpm: {score:.8}");

    let start = Instant::now();
    let mut cpm_elapsed = Duration::new(0, 0);
    for iter in 1.. {
        let updated = louvain.iterate_one_level(&network, &mut clustering);
        let cpm_start = Instant::now();
        let new_score = par_cpm(resolution, &network, &clustering);
        cpm_elapsed += cpm_start.elapsed();
        if !updated || new_score - score <= DEFAULT_EPSILON {
            score = new_score;
            break;
        }
        score = new_score;
        println!("iteration {iter} cpm: {score:.8}");
    }
    println!("final cpm: {score:.8}");

    let duration = start.elapsed();
    println!("Score time is: {cpm_elapsed:?}");
    println!("Total time is: {duration:?}");

    // Test parallel score calculation
    let score_serial = cpm(resolution, &network, &clustering);
    assert!((score - score_serial).abs() < f64::EPSILON);

    let mut res: Vec<i16> = Vec::with_capacity(n_nodes);
    let mut max_label = 0;
    for i in 0..n_nodes {
        let label = clustering.get(i) as i16;
        if max_label < label {
            max_label = label;
        }
        res.push(label);
    }
    // sort from greatest to least
    let hist = relabel_by_size(&mut res);
    println!("test cluster sizes {:?}", &hist);
    let expected = {
        let file = BufReader::new(GzDecoder::new(File::open("testdata/louvain/expected.txt.gz")?));
        let mut assn = vec![];
        for line in file.lines() {
            let line = line?;
            let mut iter = line.split_ascii_whitespace().skip(1);
            assn.push(iter.next().unwrap().parse::<i16>().unwrap());
        }
        let hist = relabel_by_size(&mut assn);
        println!("expected cluster sizes {:?}", &hist);
        assn
    };
    let ri = rand_index(&res, &expected);
    println!("rand index: {ri}");
    // Relaxed relative to the serial-louvain test above.
    // However, if you change the seed in that test you'll find that the rand index and final score vary wildly.
    assert!(1.0 - ri < 0.05);
    Ok(())
}
