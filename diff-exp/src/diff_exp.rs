//! Differential expression NB2 GLM
//!
use crate::dist;
use crate::dist::NbExactBackend;
use crate::stat::Statistics;
use anyhow::Error;
use ndarray::{arr1, Array1, Axis, Zip};
use noisy_float::types::{n64, N64};
use snoop::{CancelProgress, NoOpSnoop};
use sqz::{AdaptiveMat, MatrixMap, TransposeMap};
use std::sync::Arc;

/// Default threshold for big count (p_value computation)
pub static BIG_COUNT_DEFAULT: u64 = 900;
/// Default Zeta quintile value
pub static ZETA_QUINTILE_DEFAULT: f64 = 0.995f64;

///sseq_params struct holding global parameters for the sSeq differential expression method
#[derive(Debug, PartialEq)]
#[cfg_attr(any(feature = "serde", test), derive(serde::Deserialize))]
pub struct SSeqParams {
    /// num_cells
    pub num_cells: u32,
    /// number of genes
    pub num_genes: u32,
    /// size_factors
    pub size_factors: Vec<f64>,
    /// gene_means
    pub gene_means: Vec<f64>,
    /// gene variances
    pub gene_variances: Vec<f64>,
    /// use genes
    pub use_genes: Vec<bool>,
    /// gene moment dispersion
    pub gene_moment_phi: Vec<f64>,
    /// zeta hat
    pub zeta_hat: f64,
    /// delta
    pub delta: f64,
    /// gene dispersion
    pub gene_phi: Vec<f64>,
}

/// Result of Differential expression
#[derive(Debug)]
pub struct DiffExpResult {
    /// genes tested
    pub genes_tested: Array1<bool>,
    /// sums experiments in (cond_a)
    pub sums_in: Array1<u64>,
    /// sums experiments out (cond_b)
    pub sums_out: Array1<u64>,
    /// common means
    pub common_mean: Array1<f64>,
    /// common dispersion
    pub common_dispersion: Array1<f64>,
    /// normalized experiments in (cond_a)
    pub normalized_mean_in: Array1<f64>,
    /// normalized experiments out (cond_a)
    pub normalized_mean_out: Array1<f64>,
    /// p_values
    pub p_values: Array1<f64>,
    /// adjusted p_values (bh)
    pub adjusted_p_values: Array1<f64>,
    /// Introduce a pseudo-count into log2(fold_change)
    pub log2_fold_change: Array1<f64>,
}

pub fn sseq_differential_expression(
    mat: &AdaptiveMat,
    cond_a: &[usize],
    cond_b: &[usize],
    sseq_params: &SSeqParams,
    big_count: Option<u64>,
) -> DiffExpResult {
    let snoop = NoOpSnoop {};
    sseq_differential_expression_with_cancellation(snoop, mat, cond_a, cond_b, sseq_params, big_count).unwrap()
}

/// `sseq_differential_expression_backend`
/// Like [`sseq_differential_expression`], but lets the caller opt into a
/// [`NbExactBackend`] for the exact-test branch (e.g. the transcendental-free
/// [`NbExactBackend::Ratio`]). The default `LogSpace` reproduces
/// [`sseq_differential_expression`] exactly.
pub fn sseq_differential_expression_backend(
    mat: &AdaptiveMat,
    cond_a: &[usize],
    cond_b: &[usize],
    sseq_params: &SSeqParams,
    big_count: Option<u64>,
    backend: NbExactBackend,
) -> DiffExpResult {
    let snoop = NoOpSnoop {};
    sseq_differential_expression_with_cancellation_backend(snoop, mat, cond_a, cond_b, sseq_params, big_count, backend)
        .unwrap()
}

///sseq_differential_expression
/// Run sSeq pairwise differential expression test
pub fn sseq_differential_expression_with_cancellation(
    snoop: impl CancelProgress,
    mat: &AdaptiveMat,
    cond_a: &[usize],
    cond_b: &[usize],
    sseq_params: &SSeqParams,
    big_count: Option<u64>,
) -> Result<DiffExpResult, Error> {
    sseq_differential_expression_with_cancellation_backend(
        snoop,
        mat,
        cond_a,
        cond_b,
        sseq_params,
        big_count,
        NbExactBackend::LogSpace,
    )
}

/// `sseq_differential_expression_with_cancellation_backend`
/// Cancellation-aware sSeq pairwise differential expression with a selectable
/// [`NbExactBackend`] for the exact-test branch. This is the shared core behind
/// [`sseq_differential_expression`], [`sseq_differential_expression_backend`],
/// and [`sseq_differential_expression_with_cancellation`]. The asymptotic branch
/// (both feature-sums exceed `big_count`) is backend-independent.
pub fn sseq_differential_expression_with_cancellation_backend(
    mut snoop: impl CancelProgress,
    mat: &AdaptiveMat,
    cond_a: &[usize],
    cond_b: &[usize],
    sseq_params: &SSeqParams,
    big_count: Option<u64>,
    backend: NbExactBackend,
) -> Result<DiffExpResult, Error> {
    let big_count = big_count.unwrap_or(BIG_COUNT_DEFAULT);

    snoop.set_progress_check(0.0)?;

    let size_factor_a = cond_a.iter().fold(0., |acc, &idx| acc + sseq_params.size_factors[idx]);
    let size_factor_b = cond_b.iter().fold(0., |acc, &idx| acc + sseq_params.size_factors[idx]);
    snoop.set_progress_check(0.1)?;

    let s = snoop.get_subsnoop(0.5);
    let (feature_sums_a, feature_sums_b) = mat.sum_rows_dual_with_cancellation::<u64, _>(s, cond_a, cond_b)?;

    snoop.set_progress_check(0.6)?;

    // Per-gene NB test + BH + log2FC + normalized means from the feature sums + size factors. Owned
    // by `sseq_de_from_sums_with_cancellation` so the matrix path and Cell Ranger's batched
    // (sufficient-statistics) path share one implementation. The live `snoop` is threaded through so
    // the matrix path's 0.75/0.9/0.95/1.0 progress + cancellation checkpoints fire exactly as before.
    sseq_de_from_sums_with_cancellation(
        snoop,
        feature_sums_a.as_slice().unwrap(),
        feature_sums_b.as_slice().unwrap(),
        size_factor_a,
        size_factor_b,
        sseq_params,
        backend,
        big_count,
    )
}

/// `sseq_de_from_sums`
/// Per-gene sSeq differential expression — exact/asymptotic NB test, BH adjustment, log2 fold
/// change and normalized means — computed from precomputed per-gene feature sums and the two group
/// size-factor scalars, instead of from a matrix. This is the body of
/// [`sseq_differential_expression_with_cancellation_backend`] *after* it has reduced the matrix to
/// `feature_sums`/`size_factor_{a,b}`; that function now delegates here. Callers that produce the
/// sums by other means (e.g. Cell Ranger's shared-control batched reduction) get identical math.
///
/// This is the no-cancellation convenience wrapper (uses [`NoOpSnoop`]); use
/// [`sseq_de_from_sums_with_cancellation`] to thread progress/cancellation through. `backend` selects
/// the exact-test kernel: the matrix API defaults to [`NbExactBackend::LogSpace`]; Cell Ranger's
/// batched path passes [`NbExactBackend::Ratio`]. Reads `gene_means`, `gene_phi` and `use_genes` from
/// `params`; does **not** read `params.size_factors` (the scalars are passed directly), so a
/// [`SSeqParams`] built by [`sseq_params_from_moments`] (empty `size_factors`) works.
pub fn sseq_de_from_sums(
    feature_sums_a: &[u64],
    feature_sums_b: &[u64],
    size_factor_a: f64,
    size_factor_b: f64,
    params: &SSeqParams,
    backend: NbExactBackend,
    big_count: u64,
) -> DiffExpResult {
    let snoop = NoOpSnoop {};
    sseq_de_from_sums_with_cancellation(
        snoop,
        feature_sums_a,
        feature_sums_b,
        size_factor_a,
        size_factor_b,
        params,
        backend,
        big_count,
    )
    .unwrap()
}

/// `sseq_de_from_sums_with_cancellation`
/// Cancellation-aware [`sseq_de_from_sums`]: threads an `impl CancelProgress` through the per-gene
/// test + BH + log2FC + normalized means, firing the same `0.75 / 0.9 / 0.95 / 1.0` progress +
/// cancellation checkpoints the matrix path used before this code was extracted (the
/// `0.0 / 0.1 / 0.6` checkpoints and the `sum_rows` subsnoop stay in
/// [`sseq_differential_expression_with_cancellation_backend`]). Cell Ranger's batched caller takes
/// the [`NoOpSnoop`] default via [`sseq_de_from_sums`] and is unaffected.
#[allow(clippy::too_many_arguments)]
pub fn sseq_de_from_sums_with_cancellation(
    mut snoop: impl CancelProgress,
    feature_sums_a: &[u64],
    feature_sums_b: &[u64],
    size_factor_a: f64,
    size_factor_b: f64,
    params: &SSeqParams,
    backend: NbExactBackend,
    big_count: u64,
) -> Result<DiffExpResult, Error> {
    // compute p_value
    let mut p_values = Array1::zeros(feature_sums_a.len());

    Zip::from(&mut p_values)
        .and(feature_sums_a)
        .and(feature_sums_b)
        .and(&params.gene_means)
        .and(&params.gene_phi)
        .and(&params.use_genes)
        .par_for_each(|pv, &feature_sum_a, &feature_sum_b, &gene_mean, &gene_phi, &use_gene| {
            *pv = if use_gene && feature_sum_a > big_count && feature_sum_b > big_count {
                dist::nb_asymptotic_test(
                    feature_sum_a,
                    feature_sum_b,
                    size_factor_a,
                    size_factor_b,
                    gene_mean,
                    gene_phi,
                )
            } else {
                let exact = match backend {
                    NbExactBackend::LogSpace => dist::nb_exact_test,
                    NbExactBackend::Ratio => dist::nb_exact_test_ratio,
                };
                exact(
                    feature_sum_a,
                    feature_sum_b,
                    size_factor_a,
                    size_factor_b,
                    gene_mean,
                    gene_phi,
                )
            };
        });

    snoop.set_progress_check(0.75)?;

    // Adjust p-values for multiple testing correction
    // Only adjust the features that were actually tested
    let adj_p_values_in = p_values
        .iter()
        .enumerate()
        .filter(|(i, _)| params.use_genes[*i])
        .map(|(i, &val)| (i, val))
        .collect::<Vec<(usize, f64)>>();

    let adj_p_values = dist::adjusted_pvalue_bh(&adj_p_values_in);
    let mut p_values_bh = p_values.clone();
    for (i, adj_p_value) in adj_p_values {
        p_values_bh[i] = adj_p_value;
    }

    snoop.set_progress_check(0.9)?;

    let log2_fold_change = feature_sums_a
        .iter()
        .zip(feature_sums_b.iter())
        .map(|(&feature_sum_a, &feature_sum_b)| {
            ((1 + feature_sum_a) as f64 / (1.0 + size_factor_a)).log2()
                - ((1 + feature_sum_b) as f64 / (1.0 + size_factor_b)).log2()
        })
        .collect::<Array1<f64>>();

    snoop.set_progress_check(0.95)?;

    let sums_in = Array1::from(feature_sums_a.to_vec());
    let sums_out = Array1::from(feature_sums_b.to_vec());

    let normalized_mean_in = if size_factor_a == 0.0 {
        Array1::zeros(sums_in.dim())
    } else {
        sums_in.map(|&feature_sum_a| feature_sum_a as f64 / size_factor_a)
    };
    let normalized_mean_out = if size_factor_b == 0.0 {
        Array1::zeros(sums_out.dim())
    } else {
        sums_out.map(|&feature_sum_b| feature_sum_b as f64 / size_factor_b)
    };

    snoop.set_progress_check(1.0)?;

    Ok(DiffExpResult {
        genes_tested: arr1(&params.use_genes),
        sums_in,
        sums_out,
        common_mean: arr1(&params.gene_means),
        common_dispersion: arr1(&params.gene_phi),
        normalized_mean_in,
        normalized_mean_out,
        p_values,
        adjusted_p_values: p_values_bh,
        log2_fold_change,
    })
}

/// size_factor
fn size_factors(mat: &AdaptiveMat, cell_indices: Option<&[usize]>, umi_counts: Option<&[f64]>) -> Array1<f64> {
    let counts_per_cell = match umi_counts {
        None => match cell_indices {
            Some(cells) => mat.sum_cols::<f64>(cells),
            None => mat.sum_axis::<f64>(Axis(0)),
        },
        Some(u) => arr1(u),
    };
    let median = counts_per_cell.iter().copied().collect::<Vec<_>>().median();

    match cell_indices {
        Some(cells) => {
            let mut arr = Array1::zeros((mat.cols(),));
            for (i, &cell) in cells.iter().enumerate() {
                arr[cell] = counts_per_cell[i] / median;
            }
            arr
        }
        None => counts_per_cell.mapv(|v| v / median),
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct SizeNormalized {
    size_factors: Arc<Array1<N64>>,
}

impl SizeNormalized {
    fn new(size_factors: &Array1<f64>) -> Self {
        let size_factors = Arc::new(size_factors.mapv(|v| if v.is_nan() { n64(0.0) } else { n64(v) }));
        SizeNormalized { size_factors }
    }
}

impl MatrixMap<u32, f64> for SizeNormalized {
    type T = TransposeMap<u32, f64, SizeNormalized>;

    #[inline(always)]
    fn map(&self, v: u32, _: usize, c: usize) -> f64 {
        v as f64 / self.size_factors[c].raw()
    }

    fn t(&self) -> Self::T {
        TransposeMap::new(self.clone())
    }
}

/// `sseq_params_from_moments`
/// Compute sSeq global parameters — the method-of-moments feature-wise dispersion and its shrinkage
/// towards a high quantile — from precomputed per-gene first/second moments, instead of from a
/// matrix. This is the body of [`compute_sseq_params`] *after* it has reduced the matrix to
/// `size_factors`/`mean_g`/`var_g`; [`compute_sseq_params`] now delegates here. Callers that produce
/// the moments by other means (e.g. Cell Ranger's shared-control batched reduction) get identical
/// math.
///
/// - `sum_size_factors` is Σ(1/size_factor) over the test cells (in the matrix path,
///   Σ(1/count_total) scaled by the size-factor median).
/// - `n_cells` is the cell count m = na + nb that scales the `m·var` term.
/// - `n_genes` is the gene count, driving only the `(G-1)/(G-2)` shrinkage denominators.
///
/// The returned [`SSeqParams::size_factors`] is left empty — the moments path has no per-cell
/// size-factor vector, and the sums-based consumer [`sseq_de_from_sums`] does not read it. The
/// matrix [`compute_sseq_params`] repopulates it after delegating.
pub fn sseq_params_from_moments(
    mean_g: &[f64],
    var_g: &[f64],
    sum_size_factors: f64,
    n_cells: f64,
    n_genes: f64,
    zeta_quintile: f64,
) -> SSeqParams {
    let cell_count = n_cells;
    let gene_count = n_genes;

    // Method of moments estimate of feature-wise dispersion (phi)
    // Only use features with non-zero variance in the following estimation

    let use_g: Vec<bool> = var_g.iter().map(|&v| v > 0f64).collect();
    let mut phi_mm_g_used = Vec::<f64>::new();
    let phi_mm_g = use_g
        .iter()
        .enumerate()
        .map(|(i, &cond)| {
            let mut res: f64 = 0f64;
            if cond {
                res = 0f64.max(
                    (cell_count * var_g[i] - mean_g[i] * sum_size_factors) / (mean_g[i] * mean_g[i] * sum_size_factors),
                );

                phi_mm_g_used.push(res);
            }
            res
        })
        .collect::<Vec<f64>>();

    let (zeta_hat, delta) = if !phi_mm_g_used.is_empty() {
        //Use a high quintile of the MoM dispersion as our shrinkage target
        //# per the rule of thumb libray_tyein Yu, et al.
        let zeta_hat = phi_mm_g_used.percentile(100.0 * zeta_quintile);
        //Compute delta, the optimal shrinkage towards zeta_hat
        // This defines a linear function that shrinks the MoM dispersion estimates
        let mean_phi_mm_g = phi_mm_g_used.mean();
        let delta = (phi_mm_g_used
            .iter()
            .fold(0., |acc, &x| acc + (x - mean_phi_mm_g) * (x - mean_phi_mm_g))
            / (gene_count - 1f64))
            / (phi_mm_g_used
                .iter()
                .fold(0., |acc, &x| acc + (x - zeta_hat) * (x - zeta_hat))
                / (gene_count - 2f64));
        (zeta_hat, delta)
    } else {
        // variance of all genes was 0
        (0.0, 0.0)
    };

    // Compute the shrunken dispersion estimates
    // Interpolate between the MoM estimates and zeta_hat by delta
    let mut phi_g = vec![f64::NAN; gene_count as usize];

    let cond = phi_mm_g_used.iter().any(|x| x > &0f64);
    for (i, &v) in var_g.iter().enumerate() {
        if cond && v > 0f64 {
            phi_g[i] = (1f64 - delta) * phi_mm_g[i] + delta * zeta_hat;
        } else {
            phi_g[i] = 0f64;
        }
    }

    SSeqParams {
        size_factors: Vec::new(),
        num_cells: cell_count as u32,
        num_genes: gene_count as u32,
        gene_means: mean_g.to_vec(),
        gene_variances: var_g.to_vec(),
        use_genes: use_g,
        gene_moment_phi: phi_mm_g,
        zeta_hat,
        delta,
        gene_phi: phi_g,
    }
}

/// compute_sseq_params
pub fn compute_sseq_params(
    mat: &AdaptiveMat,
    zeta_quintile: Option<f64>,
    cell_indices: Option<&[usize]>,
    umi_counts: Option<&[f64]>,
) -> SSeqParams {
    let cell_count = cell_indices.map_or_else(|| mat.cols(), <[usize]>::len) as f64;
    let gene_count = mat.rows() as f64;
    let size_factors = size_factors(mat, cell_indices, umi_counts);
    let norm_mat = mat.view().compose_map(SizeNormalized::new(&size_factors));
    let (mean_g, var_g) = match cell_indices {
        Some(cells) => norm_mat.mean_var_rows(cells),
        None => norm_mat.mean_var_axis(Axis(1)),
    };
    let sum_size_factors = size_factors
        .iter()
        .filter(|&&v| v != 0f64)
        .fold(0f64, |acc, &v| acc + 1.0 / v);

    let mut params = sseq_params_from_moments(
        mean_g.as_slice().unwrap(),
        var_g.as_slice().unwrap(),
        sum_size_factors,
        cell_count,
        gene_count,
        zeta_quintile.unwrap_or(ZETA_QUINTILE_DEFAULT),
    );
    // The moments path leaves `size_factors` empty; the matrix path owns a per-cell size-factor
    // vector that the matrix DE (`sseq_differential_expression*`) and CR's `compute_sseq_params_o3`
    // both read back, so repopulate it here.
    params.size_factors = size_factors.to_vec();
    params
}

#[cfg(test)]
pub mod test {
    use super::*;
    use ndarray::array;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use sqz::gen_rand::random_adaptive_mat;
    #[cfg(feature = "hdf5")]
    use {crate::utils, std::path::Path};

    /// The sufficient-statistics path (`sseq_params_from_moments` + `sseq_de_from_sums`) must
    /// reproduce the matrix path (`compute_sseq_params` + `sseq_differential_expression_backend`)
    /// BYTE-IDENTICALLY (rtol=0), for both NB exact-test backends. This is the generic
    /// "sufficient-stats DE == matrix DE" guard and confirms the delegation drops no terms and that
    /// `compute_sseq_params` repopulates `size_factors` (otherwise the matrix DE below would panic).
    #[test]
    fn test_from_moments_sums_matches_matrix() {
        let n_genes: usize = 30;
        let n_cells: usize = 80;
        let range = 8u32;
        let mut rng = SmallRng::seed_from_u64(42);
        let mat = random_adaptive_mat(&mut rng, n_genes, n_cells, range, None);

        let cond_a: Vec<usize> = (0..30).collect();
        let cond_b: Vec<usize> = (30..60).collect();
        let cells: Vec<usize> = (0..60).collect();
        let zq = 0.995f64;

        // Independently extract the per-gene moments + sum_size_factors over `cells`, the same
        // reduction `compute_sseq_params` performs, so we can build params via the moments path.
        let sf = size_factors(&mat, Some(&cells), None);
        let norm = mat.view().compose_map(SizeNormalized::new(&sf));
        let (mean_g, var_g) = norm.mean_var_rows(&cells);
        let sum_size_factors = sf.iter().filter(|&&v| v != 0f64).fold(0f64, |acc, &v| acc + 1.0 / v);

        let eq = |x: f64, y: f64, ctx: &str, g: usize| {
            assert!(x == y || (x.is_nan() && y.is_nan()), "{ctx} g={g}: {x} != {y} (rtol=0)");
        };

        // Params: matrix path (delegates internally) vs moments path. Compare every field the DE
        // reads (size_factors is intentionally empty on the moments side, so skip it).
        let params_matrix = compute_sseq_params(&mat, Some(zq), Some(&cells), None);
        let params_moments = sseq_params_from_moments(
            mean_g.as_slice().unwrap(),
            var_g.as_slice().unwrap(),
            sum_size_factors,
            cells.len() as f64,
            n_genes as f64,
            zq,
        );
        assert_eq!(params_matrix.use_genes, params_moments.use_genes, "use_genes");
        eq(params_matrix.zeta_hat, params_moments.zeta_hat, "zeta_hat", 0);
        eq(params_matrix.delta, params_moments.delta, "delta", 0);
        for g in 0..n_genes {
            eq(
                params_matrix.gene_means[g],
                params_moments.gene_means[g],
                "gene_means",
                g,
            );
            eq(
                params_matrix.gene_variances[g],
                params_moments.gene_variances[g],
                "gene_var",
                g,
            );
            eq(
                params_matrix.gene_moment_phi[g],
                params_moments.gene_moment_phi[g],
                "phi_mm",
                g,
            );
            eq(params_matrix.gene_phi[g], params_moments.gene_phi[g], "gene_phi", g);
        }

        // DE: matrix path vs sums path, for both backends.
        let size_factor_a = cond_a
            .iter()
            .fold(0., |acc, &idx| acc + params_matrix.size_factors[idx]);
        let size_factor_b = cond_b
            .iter()
            .fold(0., |acc, &idx| acc + params_matrix.size_factors[idx]);
        let fa = mat.sum_rows::<u64>(&cond_a);
        let fb = mat.sum_rows::<u64>(&cond_b);

        // Pick big_count from the data (median of per-gene min(fa, fb) over tested genes) so BOTH
        // test branches fire — the exact-test branch (≤ big_count on a side) and the
        // `nb_asymptotic_test` branch (both sides > big_count) — which a fixed big_count tuned to the
        // production default (900) would silently skip on this small matrix. Asserted below.
        let mut mins: Vec<u64> = (0..n_genes)
            .filter(|&g| params_matrix.use_genes[g])
            .map(|g| fa[g].min(fb[g]))
            .collect();
        mins.sort_unstable();
        let big_count = mins[mins.len() / 2];
        let n_asymptotic = (0..n_genes)
            .filter(|&g| params_matrix.use_genes[g] && fa[g] > big_count && fb[g] > big_count)
            .count();
        assert!(
            n_asymptotic > 0,
            "no gene hit the asymptotic branch (big_count={big_count})"
        );
        assert!(
            n_asymptotic < n_genes,
            "no gene hit the exact branch (big_count={big_count})"
        );

        for backend in [NbExactBackend::LogSpace, NbExactBackend::Ratio] {
            let want =
                sseq_differential_expression_backend(&mat, &cond_a, &cond_b, &params_matrix, Some(big_count), backend);
            let got = sseq_de_from_sums(
                fa.as_slice().unwrap(),
                fb.as_slice().unwrap(),
                size_factor_a,
                size_factor_b,
                &params_moments,
                backend,
                big_count,
            );
            for g in 0..n_genes {
                assert_eq!(
                    want.genes_tested[g], got.genes_tested[g],
                    "genes_tested {backend:?} g={g}"
                );
                assert_eq!(want.sums_in[g], got.sums_in[g], "sums_in {backend:?} g={g}");
                assert_eq!(want.sums_out[g], got.sums_out[g], "sums_out {backend:?} g={g}");
                eq(want.common_mean[g], got.common_mean[g], "common_mean", g);
                eq(
                    want.common_dispersion[g],
                    got.common_dispersion[g],
                    "common_dispersion",
                    g,
                );
                eq(want.normalized_mean_in[g], got.normalized_mean_in[g], "norm_in", g);
                eq(want.normalized_mean_out[g], got.normalized_mean_out[g], "norm_out", g);
                eq(want.p_values[g], got.p_values[g], "p_values", g);
                eq(want.adjusted_p_values[g], got.adjusted_p_values[g], "adj_p", g);
                eq(want.log2_fold_change[g], got.log2_fold_change[g], "log2fc", g);
            }
        }
    }

    /// Test size factors
    #[test]
    fn test_size_factors() {
        let cols: usize = 10;
        let rows: usize = 7;
        let range = 10u32;
        let mut rng = SmallRng::seed_from_u64(0);
        let mat = random_adaptive_mat(&mut rng, rows, cols, range, None);
        let res_factor = size_factors(&mat, None, None);
        assert_eq!(array![0.0, 0.5, 2.0, 1.0, 3.0, 2.25, 2.25, 0.5, 0.5, 1.0], res_factor);
    }

    #[test]
    #[cfg(feature = "hdf5")]
    fn test_compare_sseq_differential_expression() {
        let use_umi = false;
        let min_feature_sum = None;
        let library_type = utils::GENE_EXPRESSION_LIBRARY_TYPE;
        struct TestInput<'a> {
            input_file: String,
            input_type: String,
            clustering_key: String,
            sseq_params_json: String,
            zeta_quintile: Option<f64>,
            feature_selection: Option<&'a [u32]>,
            barcode_selection: Option<&'a [usize]>,
            ground_truth_file: String,
            output_file: Option<String>,
        }
        let tests = maplit::hashmap! {
            "h5" => TestInput {
               input_file: "./test/analysis.h5".to_owned(),
               input_type: "h5".to_owned(),
               clustering_key: "_kmeans_3_clusters".to_owned(),
               sseq_params_json: String::new(),
               zeta_quintile: None,
               feature_selection: None,
               barcode_selection: None,
               ground_truth_file: String::new(),
               output_file: Some("./test/analysis_diff_exp.csv".to_string()),
            },
        };

        for (name, input) in tests {
            println!("testing {name} ....");
            utils::test::test_sseq_differential_expression(
                &input.input_type,
                Path::new(&input.input_file),
                Path::new(&input.input_file),
                library_type,
                &input.clustering_key,
                Path::new(&input.sseq_params_json),
                use_umi,
                min_feature_sum,
                input.zeta_quintile,
                input.feature_selection,
                input.barcode_selection,
                Path::new(&input.ground_truth_file),
                input.output_file.as_ref().map(Path::new),
            )
            .unwrap();
        }
    }

    // TODO: restore this test
    // #[test]
    // fn test_compare_sseq_diffexp_submatrix() -> Result<(), Error> {
    //     let use_umi = false;
    //     let min_feature_sum = Some(f64::MIN_POSITIVE);
    //     let library_type = utils::GENE_EXPRESSION_LIBRARY_TYPE;
    //     let matrix_path = "./test/test2.cloupe";
    //     let barcode_selection = Some(
    //         &[
    //             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 25, 26, 27, 28, 29, 30, 32, 33,
    //             34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 63,
    //             64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91,
    //             92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    //         ][..],
    //     );
    //     let test_cluster = 0;

    //     let (mat, sseq_params, _) = utils::init_matrix(
    //         "loupe",
    //         &Path::new(&matrix_path),
    //         library_type,
    //         use_umi,
    //         min_feature_sum,
    //         None,
    //         None,
    //         barcode_selection,
    //     )?;

    //     utils::test::compare_sseq_params(
    //         &Path::new("./test/c1+2.sseq_params.json.gz"),
    //         &sseq_params,
    //         barcode_selection,
    //     );

    //     let clusters = {
    //         let mut reader = CLoupeReader::new(&matrix_path)?;
    //         let section = reader.get_immutable_section()?;
    //         let cluster = section.clusterings.iter().find(|&c| c.name == "Graph-based");
    //         read_clustering_assignments(&reader, cluster.unwrap())?
    //     };

    //     let (cond_a, cond_b) = {
    //         let mut cond_a = vec![];
    //         let mut cond_b = vec![];
    //         for (i, &cluster) in clusters.iter().enumerate() {
    //             if cluster == test_cluster {
    //                 cond_a.push(i);
    //             } else if cluster == test_cluster + 1 {
    //                 cond_b.push(i);
    //             }
    //         }
    //         (cond_a, cond_b)
    //     };

    //     let de_result = sseq_differential_expression(&mat.matrix, &cond_a, &cond_b, &sseq_params, None);
    //     let deg_map = utils::test::DegResult::read_deg_result("./test/c1+2.diffexp.csv.gz", test_cluster);

    //     utils::test::CompareResult::diff_result(None, deg_map, &de_result, &mat);
    //     Ok(())
    // }
}
