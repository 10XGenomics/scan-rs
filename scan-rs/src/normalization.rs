use crate::stats::median_mut;
use anyhow::{bail, Error};
use ndarray::prelude::*;
use sqz::{AdaptiveMat, AdaptiveVec, LowRankOffset, MatrixMap, ScaleAxis, TransposeMap};
use std::f64;
use std::ops::Deref;
use std::str::FromStr;

/// Normalization scheme for feature-barcode UMI count matrix
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Normalization {
    /// Cell Ranger style log normalization. Barcode UMI count totals are scaled to the median
    /// and the transformation `x -> log2(1 + x)` is applied
    CellRanger,
    /// Cell Ranger style log normalization, like above, just minus the variance standardization
    CellRanger8,
    /// Seurat style log normalization (pre SCTransform). Barcode UMI count totals are scaled
    /// to 10,000 and the transformation `x -> ln(1 + x)` is applied
    SeuratLog,
    /// binomial deviance residuals
    BinomialDeviance,
    /// binomial Pearson residuals
    BinomialPearson,
    /// size factors specified explicity
    WithSizeFactors,
    /// vanilla log 2
    LogTransform,
}

impl FromStr for Normalization {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cellranger" => Ok(Normalization::CellRanger),
            "cellranger8" => Ok(Normalization::CellRanger8),
            "seuratlog" => Ok(Normalization::SeuratLog),
            "binomialdeviance" => Ok(Normalization::BinomialDeviance),
            "binomialpearson" => Ok(Normalization::BinomialPearson),
            _ => bail!("Normalization not recognized: {}", s),
        }
    }
}

/// Normalize an `AdaptiveMat` of `u32` and return a `LowRankOffset`
pub fn normalize<D, M>(mat: AdaptiveMat<u32, D, M>, norm: Normalization) -> LowRankOffset<D, impl MatrixMap<u32, f64>>
where
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, u32>,
{
    use Normalization::{CellRanger, CellRanger8, SeuratLog};
    match norm {
        CellRanger => log_normalize(mat, None, LogBase::Two).scale_and_center(Axis(1), None),
        CellRanger8 => {
            // TODO: fix the opaque types issue and this can become .center(Axis(1), None)
            let scale = Array1::<f64>::ones((mat.rows(),));
            log_normalize(mat, None, LogBase::Two).scale_and_center(Axis(1), Some(scale))
        }
        SeuratLog => log_normalize(mat, Some(10_000_f64), LogBase::E).scale_and_center(Axis(1), None),
        _ => panic!("not implemented"),
        // TODO: These functions return a different opaque type, so this is going to
        //   require some factoring. We may want to make a trait that covers a lot of
        //   LowRankOffset’s API (rows, cols, t, shape), and require that it implement
        //   Dot<ArrayBase<DS, Ix2>>, and ArrayBase<DS, Ix2>: Dot<LowRankOffset<D, M>>,
        //   and return a Box<AdaptiveMat<_, D, M>>
        // BinomialDeviance => binom_deviance_resid(mat),
        // BinomPearson => binom_pearson_resid(mat),
    }
}

/// Log normalize an `AdaptiveMat` of `u32` by a size factor  and return a `LowRankOffset`
pub fn normalize_with_size_factor<D, M>(
    mat: AdaptiveMat<u32, D, M>,
    norm: Normalization,
    size_factors: Option<Array1<u32>>,
) -> LowRankOffset<D, impl MatrixMap<u32, f64>>
where
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, u32>,
{
    use Normalization::{CellRanger, CellRanger8, LogTransform, SeuratLog, WithSizeFactors};
    match norm {
        CellRanger => log_normalize_with_size_factor(mat, None, LogBase::Two, None).scale_and_center(Axis(1), None),
        CellRanger8 => {
            // TODO: fix the opaque types issue and this can become .center(Axis(1), None)
            let scale = Array1::<f64>::ones((mat.rows(),));
            log_normalize_with_size_factor(mat, None, LogBase::Two, None).scale_and_center(Axis(1), Some(scale))
        }
        SeuratLog => {
            log_normalize_with_size_factor(mat, Some(10_000_f64), LogBase::E, None).scale_and_center(Axis(1), None)
        }
        WithSizeFactors => {
            log_normalize_with_size_factor(mat, None, LogBase::Two, size_factors).scale_and_center(Axis(1), None)
        }
        LogTransform => {
            // Size factor of all ones and target umi count of one implies that there is no scaling before taking the log
            let all_ones = Array1::<u32>::ones((mat.cols(),));
            log_normalize_with_size_factor(mat, Some(1.0), LogBase::Two, Some(all_ones)).scale_and_center(Axis(1), None)
        }
        _ => panic!("not implemented"),
    }
}

/// Base of logarithm used by log_normalize
pub enum LogBase {
    /// ln
    E,
    /// log2
    Two,
    /// log10
    Ten,
}

/// Log-normalize an `AdaptiveMat` of `u32` and return a `LowRankOffset`:
/// 1. Scale each column (barcode) to have the same total UMI count given by
///    `umi_count_sum`. If `umi_count_sum` is `None`, use the median total UMI count
/// 2. Apply a transform `x -> log_b(1 + x)`, with `b` specified by `log_base`
/// 3. Center and scale each row (feature) to mean 0 and variance 1
pub fn log_normalize<D, M>(
    matrix: AdaptiveMat<u32, D, M>,
    umi_count_sum: Option<f64>,
    log_base: LogBase,
) -> AdaptiveMat<f64, D, impl MatrixMap<u32, f64>>
where
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, u32>,
{
    log_normalize_with_size_factor(matrix, umi_count_sum, log_base, None)
}

/// Log-normalize an `AdaptiveMat` of `u32` and return a `LowRankOffset`:
/// 1. Scale each column (barcode) to have the same total UMI count given by
///    `umi_count_sum`* `size_factors[i]` for the i-th column.
///     If `umi_count_sum` is `None`, use the median total UMI count
///     If `size_factors` uses the total UMI counts in each column
/// 2. Apply a transform `x -> log_b(1 + x)`, with `b` specified by `log_base`
/// 3. Center and scale each row (feature) to mean 0 and variance 1
pub fn log_normalize_with_size_factor<D, M>(
    matrix: AdaptiveMat<u32, D, M>,
    umi_count_sum: Option<f64>,
    log_base: LogBase,
    size_factors: Option<Array1<u32>>,
) -> AdaptiveMat<f64, D, impl MatrixMap<u32, f64>>
where
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, u32>,
{
    let normalization_counts = match size_factors {
        Some(x) => {
            assert_eq!(
                x.len(),
                matrix.cols(),
                "Size of the size factor and matrix columns dont match.\nSize factor length {}; Matrix dimensions: {:?}",
                &x.len(),
                &matrix.shape()
            );
            x
        }
        None => matrix.sum_axis(Axis(0)),
    };
    let mut umi_counts = matrix.sum_axis(Axis(0));
    let target_umi_count: f64 = match umi_count_sum {
        Some(x) => x,
        None => {
            // Scale to the median UMI count. Clone because median_mut will reorder its argument.
            median_mut(&mut umi_counts).map_or(1.0, |median: u32| (median as f64).max(1.0))
        }
    };
    let col_scales = normalization_counts.mapv(|c: u32| target_umi_count / (c as f64));
    let scale_cols = ScaleAxis::new(Axis(1), col_scales);

    let log1p_fn = match log_base {
        LogBase::E => |x: f64| (x + 1.0).ln(),
        LogBase::Two => |x: f64| (x + 1.0).log2(),
        LogBase::Ten => |x: f64| (x + 1.0).log10(),
    };
    matrix.compose_map(scale_cols).apply(log1p_fn)
}

/// Fit the null multinomial model for normalization based on deviance/Pearson residuals.
/// Returns a tuple `(n, p)` where `n` is an array of total barcode counts and `p` is an
/// array of feature abundances.
fn fit_multinomial_model<D, M>(matrix: &AdaptiveMat<u32, D, M>) -> (Array1<f64>, Array1<f64>)
where
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, u32>,
{
    let n = matrix.sum_axis::<f64>(Axis(0));
    let total_umi_count = n.sum();
    let pi = matrix.sum_axis::<f64>(Axis(1)).mapv_into(|x| x / total_umi_count);
    (n, pi)
}

/// Normalize an `AdaptiveMat` of `u32` according to the binomial deviance approximation to
/// multinomial GLM-PCA (Townes, et al. 2019; https://doi.org/10.1186/s13059-019-1861-6).
/// Uses a `LowRankOffset` to avoid densifying the matrix.
pub fn binom_deviance_resid<D, M>(matrix: AdaptiveMat<u32, D, M>) -> LowRankOffset<D, impl MatrixMap<u32, f64>>
where
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, u32>,
{
    // y_fb = UMI count for barcode b, feature f
    // n_b = sum_f y_fb = total UMI counts from barcode b
    // pi_f = (sum_b y_fb) / (sum_fb y_fb) = fraction of total UMI counts from feature f
    // mu_fb = n_b pi_f = expectation of y_fb under null model
    //
    // Deviance residual:
    //   r_fb = sign(y_fb - mu_fb) sqrt(2 y_fb log(y_fb/mu_fb) + 2 (n_b - y_fb) log((n_b - y_fb) / (n_b - mu_fb) )
    // Pearson residual:
    //   r_fb = (y_fb - mu_fb) / sqrt( mu_fb - mu_fb^2 / n_b )
    let (n, pi) = fit_multinomial_model(&matrix);

    let u = pi
        .mapv(|x| (1.0 / (1.0 - x)).ln().sqrt())
        .into_shape((matrix.rows(), 1))
        .unwrap();
    let v = n.mapv(|x| -(2.0 * x).sqrt()).into_shape((1, matrix.cols())).unwrap();

    let dev_resid_map = BinomDevMap { n, pi };

    LowRankOffset::new(matrix.set_map(dev_resid_map), u, v)
}

#[derive(Clone, Debug, PartialEq)]
struct BinomDevMap {
    n: Array1<f64>,
    pi: Array1<f64>,
}

/// 0 log(0) = 0
#[inline]
fn a_ln_a_over_b(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        return 0.0;
    }
    a * (a / b).ln()
}

impl MatrixMap<u32, f64> for BinomDevMap {
    type T = TransposeMap<u32, f64, Self>;

    #[inline(always)]
    fn map(&self, v: u32, r: usize, c: usize) -> f64 {
        let v = v as f64;

        let n = self.n[c];
        let pi = self.pi[r];
        let mu = n * pi;

        let sign = (v - mu).signum();
        // NOTE: x.max(0) needed before sqrt due to floating point errors
        let residual = sign
            * (2.0 * (a_ln_a_over_b(v, mu) + a_ln_a_over_b(n - v, n - mu)))
                .max(0.0)
                .sqrt();

        let zero_term = -((2.0 * n * (1.0 / (1.0 - pi)).ln()).sqrt());

        residual - zero_term
    }

    fn t(&self) -> Self::T {
        TransposeMap::new(self.clone())
    }
}

impl Eq for BinomDevMap {}

/// Normalize an `AdaptiveMat` of `u32` according to the binomial Pearson residual approximation to
/// multinomial GLM-PCA (Townes, et al. 2019; https://doi.org/10.1186/s13059-019-1861-6).
/// Uses a `LowRankOffset` to avoid densifying the matrix.
pub fn binom_pearson_resid<D, M>(matrix: AdaptiveMat<u32, D, M>) -> LowRankOffset<D, impl MatrixMap<u32, f64>>
where
    D: Deref<Target = [AdaptiveVec]>,
    M: MatrixMap<u32, u32>,
{
    let (n, pi) = fit_multinomial_model(&matrix);

    let u = pi
        .mapv(|x| (x / (1.0 - x)).sqrt())
        .into_shape((matrix.rows(), 1))
        .unwrap();
    let v = n.mapv(|x| -x.sqrt()).into_shape((1, matrix.cols())).unwrap();

    let pearson_resid_map = BinomPearsonMap { n, pi };

    LowRankOffset::new(matrix.set_map(pearson_resid_map), u, v)
}

#[derive(Clone, Debug, PartialEq)]
struct BinomPearsonMap {
    n: Array1<f64>,
    pi: Array1<f64>,
}

impl MatrixMap<u32, f64> for BinomPearsonMap {
    type T = TransposeMap<u32, f64, Self>;

    #[inline(always)]
    fn map(&self, v: u32, r: usize, c: usize) -> f64 {
        let v = v as f64;

        let n = self.n[c];
        let pi = self.pi[r];
        let mu = n * pi;

        let residual = (v - mu) / (mu * (1.0 - pi)).sqrt();
        let zero_term = -(n * pi / (1.0 - pi)).sqrt();

        residual - zero_term
    }

    fn t(&self) -> Self::T {
        TransposeMap::new(self.clone())
    }
}

impl Eq for BinomPearsonMap {}

#[cfg(test)]
mod test_normalization {

    use super::*;
    use crate::dim_red::bk_svd::BkSvd;
    use crate::dim_red::Pca;
    use anyhow::Error;
    use ndarray::Array1;
    use sqz::AdaptiveMatOwned;

    #[test]
    /// This is a 'manual' verification that the Cellranger-style matrix-preprocessing and PCA
    /// is reproduced in this crate. We take the correct singular values of the preprocessed matrix out of
    /// cellranger (using an SVD, not irlba), and make sure we get the same values here.
    /// To get the 'correct' singular values, we patch pca.py in Cell Ranger to do an full svd of the
    /// matrix that is passed to irlb(), with the centering and scaling applied.  
    /// Here we test that the Scaling::cellranger + RandPca recover the same singular values.
    fn test_cellranger_normalization() -> Result<(), Error> {
        let matrix = crate::mtx::load_mtx("test/pbmc4k_tiny.mtx.gz")?;
        let (matrix, _, _, _) = matrix.partition_on_threshold(3.0);

        let norm_mat = normalize(matrix.view(), Normalization::CellRanger);
        let (_, d, _) = BkSvd::new().run_pca(&norm_mat, 10).unwrap();

        let correct_d = Array1::from(vec![
            158.15242333,
            125.37151293,
            113.40948418,
            109.22340589,
            105.67964003,
            104.21194153,
            103.00044884,
            102.08798615,
            101.14301241,
            99.65312695,
        ]);

        if !d.abs_diff_eq(&correct_d, 0.05) {
            println!("correct d: {correct_d:?}");
            println!("observed d: {d:?}");
            panic!("cellranger-style normalization isn't matching CR 4.0");
        }

        Ok(())
    }

    #[test]
    fn test_binom_deviance() -> Result<(), Error> {
        let matrix = crate::mtx::load_mtx("test/pbmc4k_tiny.mtx.gz")?;
        let (matrix, _, _, _) = matrix.partition_on_thresholds(Some(3.0), None);

        println!("{:?}", matrix.shape());
        let norm_mat = binom_deviance_resid(matrix.view());
        let (_, d, _) = BkSvd::new().run_pca(&norm_mat, 10)?;

        // library(Matrix)
        // mat <- Matrix::readMM("~/git/scan-rs/scan-rs/test/pbmc4k_tiny.mtx.gz")
        // mat <- mat[Matrix::rowSums(mat) >= 3,]
        // source("~/git/scrna2019/util/functions.R") # github.com/willtownes/scrna2019.git
        // resid <- null_residuals(mat, mod = 'binomial', type = 'deviance')
        // svd(resid)$d[1:10]
        // [1] 231.49399 183.69837 124.92083 117.03172 104.63398  95.51693  89.26010  80.07947  78.67546  76.05272
        let correct_d = Array1::from(vec![
            231.49399, 183.69837, 124.92083, 117.03172, 104.63398, 95.51693, 89.26010, 80.07947, 78.67546, 76.05272,
        ]);

        if !d.abs_diff_eq(&correct_d, 0.05) {
            println!("correct d: {correct_d:?}");
            println!("observed d: {d:?}");
            panic!("binomial deviance residual normalization isn't matching results from Townes et al.");
        }

        Ok(())
    }

    #[test]
    fn test_binom_pearson() -> Result<(), Error> {
        let matrix = crate::mtx::load_mtx("test/pbmc4k_tiny.mtx.gz")?;
        let (matrix, _, _, _) = matrix.partition_on_thresholds(Some(3.0), None);

        println!("{:?}", matrix.shape());
        let norm_mat = binom_pearson_resid(matrix.view());
        let (_, d, _) = BkSvd::new().run_pca(&norm_mat, 10)?;

        // library(Matrix)
        // mat <- Matrix::readMM("~/git/scan-rs/scan-rs/test/pbmc4k_tiny.mtx.gz")
        // mat <- mat[Matrix::rowSums(mat) >= 3,]
        // source("~/git/scrna2019/util/functions.R") # github.com/willtownes/scrna2019.git
        // resid <- null_residuals(mat, mod = 'binomial', type = 'pearson')
        // svd(resid)$d[1:10]
        // [1] 233.9072 193.7451 163.9111 150.8670 137.7290 124.7651 114.8776 113.0533 112.0241 106.6563
        let correct_d = Array1::from(vec![
            233.9072, 193.7451, 163.9111, 150.8670, 137.7290, 124.7651, 114.8776, 113.0533, 112.0241, 106.6563,
        ]);

        if !d.abs_diff_eq(&correct_d, 0.05) {
            println!("correct d: {correct_d:?}");
            println!("observed d: {d:?}");
            panic!("binomial Pearson residual normalization isn't matching results from Townes et al.");
        }

        Ok(())
    }

    #[test]
    fn test_fit_multinomial_model() {
        let mat: Array2<u32> = array![[1, 0, 2], [0, 0, 0], [3, 0, 6]];

        // column sums
        let expected_n = array![4.0, 0.0, 8.0];
        let expected_pi = array![0.25, 0.0, 0.75];

        let adaptive_mat = AdaptiveMatOwned::<f64>::from_dense(mat.view());
        let (n, pi) = fit_multinomial_model(&adaptive_mat);

        assert_close(n.view(), expected_n.view());
        assert_close(pi.view(), expected_pi.view());
    }

    #[test]
    fn test_one_dim() {
        let mat: Array2<u32> = Array2::from_shape_vec(
            (1, 649),
            vec![
                225, 44, 63, 17, 59, 565, 77, 169, 108, 46, 17, 46, 67, 626, 35, 51, 96, 355, 5, 675, 48, 17, 267, 40,
                51, 74, 17, 67, 23, 126, 59, 110, 137, 71, 191, 42, 59, 54, 81, 201, 49, 25, 21, 221, 200, 159, 13, 33,
                19, 62, 71, 19, 54, 24, 55, 22, 34, 96, 8, 8, 195, 503, 218, 184, 89, 5, 34, 21, 298, 28, 21, 330, 515,
                110, 119, 403, 128, 9, 4, 60, 23, 148, 60, 177, 121, 141, 61, 241, 221, 168, 46, 57, 296, 85, 36, 78,
                96, 17, 79, 87, 22, 49, 121, 219, 285, 53, 53, 217, 251, 83, 545, 344, 27, 155, 60, 195, 298, 235, 86,
                110, 216, 54, 247, 26, 63, 13, 99, 29, 89, 121, 118, 59, 47, 32, 114, 127, 87, 20, 153, 142, 80, 36,
                27, 254, 7, 29, 240, 167, 295, 119, 67, 11, 55, 22, 97, 14, 118, 13, 18, 9, 81, 304, 108, 24, 44, 90,
                87, 34, 103, 264, 192, 127, 99, 110, 99, 138, 138, 54, 128, 157, 72, 56, 323, 162, 84, 102, 557, 82,
                65, 61, 81, 124, 303, 71, 88, 63, 177, 284, 30, 48, 176, 35, 24, 271, 122, 69, 159, 229, 209, 36, 7,
                169, 78, 10, 182, 59, 15, 128, 80, 68, 59, 97, 161, 114, 67, 33, 274, 221, 73, 44, 101, 253, 355, 99,
                57, 95, 115, 102, 168, 423, 456, 93, 20, 69, 440, 325, 132, 165, 5, 22, 94, 40, 175, 305, 102, 31, 13,
                30, 218, 38, 43, 45, 23, 73, 25, 13, 39, 49, 5, 32, 250, 28, 58, 18, 207, 84, 45, 34, 39, 272, 39, 474,
                68, 132, 61, 129, 87, 22, 5, 67, 62, 13, 36, 243, 328, 33, 13, 62, 165, 94, 157, 248, 145, 107, 9, 170,
                56, 119, 84, 54, 25, 86, 120, 4, 134, 73, 178, 40, 24, 639, 60, 26, 47, 4, 207, 24, 87, 172, 535, 80,
                56, 47, 71, 151, 82, 7, 25, 8, 44, 18, 333, 67, 127, 344, 61, 108, 66, 95, 141, 7, 6, 60, 66, 114, 349,
                263, 7, 57, 96, 364, 54, 219, 51, 165, 202, 77, 265, 29, 35, 62, 40, 44, 45, 3, 58, 3, 165, 254, 175,
                50, 64, 90, 58, 112, 46, 22, 29, 42, 215, 292, 64, 9, 17, 56, 20, 14, 52, 75, 51, 44, 154, 75, 44, 64,
                90, 96, 353, 118, 136, 196, 49, 119, 126, 34, 35, 225, 34, 53, 9, 54, 96, 83, 28, 44, 34, 120, 56, 217,
                50, 36, 39, 353, 64, 47, 99, 34, 538, 60, 16, 36, 41, 28, 81, 261, 43, 26, 133, 65, 9, 16, 8, 59, 16,
                869, 26, 97, 47, 56, 25, 7, 61, 67, 51, 106, 289, 95, 164, 72, 94, 61, 102, 287, 15, 37, 9, 36, 53,
                180, 43, 97, 155, 70, 26, 413, 154, 75, 53, 10, 15, 113, 283, 45, 71, 73, 103, 90, 84, 75, 47, 23, 151,
                59, 34, 76, 114, 97, 268, 220, 54, 125, 74, 106, 86, 42, 47, 4, 160, 123, 3, 35, 65, 78, 125, 4, 13,
                137, 264, 16, 11, 27, 24, 89, 28, 311, 104, 95, 45, 59, 97, 35, 113, 91, 92, 46, 25, 302, 172, 534,
                157, 45, 247, 90, 353, 79, 273, 60, 395, 245, 118, 227, 62, 37, 35, 64, 437, 8, 122, 145, 131, 93, 41,
                122, 70, 156, 401, 62, 665, 83, 37, 314, 117, 103, 8, 262, 85, 48, 96, 125, 41, 137, 61, 30, 75, 80,
                42, 85, 89, 148, 92, 88, 23, 85, 95, 112, 168, 165, 39, 78, 74, 15, 97, 146, 35, 7, 11, 67, 3, 44, 34,
                254, 131, 50, 213, 94, 29, 17, 173, 66, 106, 109, 121, 50, 40, 21, 20, 19, 10, 17, 23, 30, 62, 30, 18,
                31, 62, 211, 32, 103, 92,
            ],
        )
        .unwrap();
        let adaptive_mat = AdaptiveMatOwned::<u32>::from_dense(mat.view());
        let norm_mat = normalize(adaptive_mat.view(), Normalization::CellRanger).t().to_dense();
        assert!(!norm_mat.fold(false, |acc, x| x.is_nan() || acc));
    }

    //use super::*;
    //use ndarray::{ArrayView, Dimension};

    // #[test]
    // fn pcard_test1() {
    //     let (_mat, mat, zeroed_features) = crate::dim_red::test::cr_test_zeroed_matrix(5, 6, 0.4, 2);
    //     println!("mat: {:?}", mat);
    //     let mat: AdaptiveMat<(), Vec<AdaptiveVec>> = AdaptiveMat::<(), Vec<AdaptiveVec>>::from_dense(mat.view());

    //     test_pcard_mat(&mat);
    // }

    // fn test_pcard_mat(mat: &AdaptiveMat) {
    //     // make the mapper version of pcard
    //     let pcard = make_pcard_mapper(&mat);

    //     println!("pcard: {:?}", pcard);

    //     let low_rank_pcard = pca_rd_mat(mat.clone());
    //     let low_rank_to_dense = low_rank_pcard.to_dense();

    //     // make the dense version of the input matrix
    //     let d = mat.to_dense();

    //     // map dense matrix to pcard values, with independent impl
    //     let mut dense_explicit = ndarray::Array2::zeros((mat.rows(), mat.cols()));

    //     for (coord, value) in d.indexed_iter() {
    //         let (feature, cell) = coord;
    //         let y = *value as f64;

    //         let n = pcard.n[cell];
    //         let mu = pcard.n[cell] * pcard.pi[feature];
    //         let pi_j = pcard.pi[feature];

    //         let sign = if (y - mu) > 0.0 { 1.0 } else { -1.0 };
    //         let term2 = 2.0 * a_ln_a_over_b(y, mu) + 2.0 * a_ln_a_over_b(n - y, n - mu);

    //         dense_explicit[coord] = sign * term2.sqrt();
    //     }

    //     println!("explicit: {:?}", dense_explicit);
    //     println!("lr: {:?}", low_rank_to_dense);

    //     assert_close(low_rank_to_dense.view(), dense_explicit.view());
    // }

    // stolen from ndarray - not currently exported.
    fn assert_close<D>(a: ArrayView<f64, D>, b: ArrayView<f64, D>)
    where
        D: Dimension,
    {
        let diff = (&a - &b).mapv_into(f64::abs);

        let rtol = 1e-7;
        let atol = 1e-12;
        let crtol = b.mapv(|x| x.abs() * rtol);
        let tol = crtol + atol;
        let tol_m_diff = &diff - &tol;
        let maxdiff = tol_m_diff.fold(f64::NAN, |x, y| f64::max(x, *y));
        println!("diff offset from tolerance level= {maxdiff:.2e}");
        if maxdiff > 0. || maxdiff.is_nan() {
            println!("{a:.4?}");
            println!("{b:.4?}");
            panic!("results differ");
        }
    }

    #[test]
    fn test_cellranger_normalisation() {
        // # Python code to reconstruct this test
        // mat = np.array([[136, 936, 0, 0, 264],
        //     [134, 682, 417, 8, 391],
        //     [0, 133, 780, 0, 0],
        //     [396, 76, 96, 198, 0],
        //         ])
        // scale_factor = mat.sum(axis=0)
        // target_umi_count = np.median(mat.sum(axis=0))
        // half_processed_mat = mat.dot(np.diag(target_umi_count/scale_factor))
        // almost_processed_mat = np.log2(1 + half_processed_mat)
        // centering_factor = almost_processed_mat.mean(axis = 1).reshape((4,1))
        // scaling_factor = 1/np.std(almost_processed_mat, axis=1)
        // norm_mat = np.diag(scaling_factor).dot(almost_processed_mat - centering_factor)
        // norm_mat
        // array([[ 0.61392149,  0.95459951, -1.21707302, -1.21707302,  0.86562504],
        //    [-0.11878431,  0.54279925,  0.38607315, -1.85660965,  1.04652156],
        //    [-0.78758751,  0.76437149,  1.59839105, -0.78758751, -0.78758751],
        //    [ 0.88718256, -0.25584717, -0.01048423,  1.09574143, -1.71659259]])

        let dense: Array2<u32> = array![
            [136, 936, 0, 0, 264],
            [134, 682, 417, 8, 391],
            [0, 133, 780, 0, 0],
            [396, 76, 96, 198, 0],
        ];
        let expected_out_dense = array![
            [0.61392149, 0.95459951, -1.21707302, -1.21707302, 0.86562504],
            [-0.11878431, 0.54279925, 0.38607315, -1.85660965, 1.04652156],
            [-0.78758751, 0.76437149, 1.59839105, -0.78758751, -0.78758751],
            [0.88718256, -0.25584717, -0.01048423, 1.09574143, -1.71659259]
        ];
        let mtx = AdaptiveMatOwned::<u32>::from_dense(dense.view());
        let norm_mat = normalize_with_size_factor(mtx, Normalization::CellRanger, None);

        assert!(expected_out_dense.abs_diff_eq(&norm_mat.to_dense(), 1e-6));
    }

    #[test]
    fn test_cellranger8_normalisation() {
        // # Python code to reconstruct this test
        // mat = np.array([[136, 936, 0, 0, 264],
        //     [134, 682, 417, 8, 391],
        //     [0, 133, 780, 0, 0],
        //     [396, 76, 96, 198, 0],
        //         ])
        // scale_factor = mat.sum(axis=0)
        // target_umi_count = np.median(mat.sum(axis=0))
        // half_processed_mat = mat.dot(np.diag(target_umi_count/scale_factor))
        // almost_processed_mat = np.log2(1 + half_processed_mat)
        // centering_factor = almost_processed_mat.mean(axis = 1).reshape((4,1))
        // norm_mat = almost_processed_mat - centering_factor
        // norm_mat
        // array([[ 2.37992764,  3.70059981, -4.71810445, -4.71810445,  3.35568145],
        // [-0.15920674,  0.72751443,  0.51745426, -2.48841594,  1.40265399],
        // [-2.85652852,  2.77232551,  5.79726005, -2.85652852, -2.85652852],
        // [ 2.94151467, -0.84827885, -0.0347612 ,  3.63300591, -5.69148053]])

        let dense: Array2<u32> = array![
            [136, 936, 0, 0, 264],
            [134, 682, 417, 8, 391],
            [0, 133, 780, 0, 0],
            [396, 76, 96, 198, 0],
        ];
        let expected_out_dense = array![
            [2.37992764, 3.70059981, -4.71810445, -4.71810445, 3.35568145],
            [-0.15920674, 0.72751443, 0.51745426, -2.48841594, 1.40265399],
            [-2.85652852, 2.77232551, 5.79726005, -2.85652852, -2.85652852],
            [2.94151467, -0.84827885, -0.0347612, 3.63300591, -5.69148053]
        ];
        let mtx = AdaptiveMatOwned::<u32>::from_dense(dense.view());
        let norm_mat = normalize_with_size_factor(mtx, Normalization::CellRanger8, None);

        assert!(expected_out_dense.abs_diff_eq(&norm_mat.to_dense(), 1e-6));
    }

    #[test]
    fn test_log_normalize_with_size_factor() {
        // # Python code to reconstruct this test
        // mat = np.array([[136, 936, 0, 0, 264],
        //     [134, 682, 417, 8, 391],
        //     [0, 133, 780, 0, 0],
        //     [396, 76, 96, 198, 0],
        // ])
        // features_picked = [0, 2]
        // scale_factor = 1 + mat[features_picked,:].sum(axis=0)
        // target_umi_count = np.median(mat.sum(axis=0))
        // half_processed_mat = mat.dot(np.diag(target_umi_count/scale_factor))
        // processed_mat = np.log2(1 + half_processed_mat)
        // processed_mat
        // array([[ 9.37098961,  9.18882221,  0.        ,  0.        ,  9.37609671],
        //     [ 9.34964848,  8.73300582,  8.4781546 , 12.37964912,  9.94202202],
        //     [ 0.        ,  6.3885887 ,  9.3796973 ,  0.        ,  0.        ],
        //     [10.91145213,  5.59409085,  6.37267837, 17.00874593,  0.        ]])

        let dense: Array2<u32> = array![
            [136, 936, 0, 0, 264],
            [134, 682, 417, 8, 391],
            [0, 133, 780, 0, 0],
            [396, 76, 96, 198, 0],
        ];
        let expected_out_dense = array![
            [9.37098961, 9.18882221, 0., 0., 9.37609671],
            [9.34964848, 8.73300582, 8.4781546, 12.37964912, 9.94202202],
            [0., 6.3885887, 9.3796973, 0., 0.],
            [10.91145213, 5.59409085, 6.37267837, 17.00874593, 0.]
        ];
        let mtx = AdaptiveMatOwned::<u32>::from_dense(dense.view());
        let features_picked = [0, 2];
        let size_factors = 1 + mtx.select_rows(&features_picked).sum_axis::<u32>(Axis(0));
        let processed_mtx = log_normalize_with_size_factor(mtx, None, LogBase::Two, Some(size_factors));

        assert!(expected_out_dense.abs_diff_eq(&processed_mtx.to_dense(), 1e-6));
    }

    #[test]
    fn test_vanilla_log_norm() {
        // # Python code to reconstruct this test
        // mat = np.array([[136, 936, 0, 0, 264],
        //         [134, 682, 417, 8, 391],
        //         [0, 133, 780, 0, 0],
        //         [396, 76, 96, 198, 0],
        //  ])
        // almost_processed_mat = np.log2(1 + mat)
        // centering_factor = almost_processed_mat.mean(axis = 1).reshape((4,1))
        // scaling_factor = 1/np.std(almost_processed_mat, axis=1)
        // norm_mat = np.diag(scaling_factor).dot(almost_processed_mat - centering_factor)
        // norm_mat
        // array([[ 0.50075509,  1.16407001, -1.1965938 , -1.1965938 ,  0.72836249],
        //     [-0.14245194,  0.89844192,  0.58318993, -1.88113806,  0.54195815],
        //     [-0.80111703,  0.89623633,  1.50711477, -0.80111703, -0.80111703],
        //     [ 0.92609909,  0.14507504,  0.25503138,  0.59722303, -1.92342854]])

        let dense: Array2<u32> = array![
            [136, 936, 0, 0, 264],
            [134, 682, 417, 8, 391],
            [0, 133, 780, 0, 0],
            [396, 76, 96, 198, 0],
        ];
        let expected_out_dense = array![
            [0.50075509, 1.16407001, -1.1965938, -1.1965938, 0.72836249],
            [-0.14245194, 0.89844192, 0.58318993, -1.88113806, 0.54195815],
            [-0.80111703, 0.89623633, 1.50711477, -0.80111703, -0.80111703],
            [0.92609909, 0.14507504, 0.25503138, 0.59722303, -1.92342854]
        ];
        let mtx = AdaptiveMatOwned::<u32>::from_dense(dense.view());
        let norm_mat = normalize_with_size_factor(mtx, Normalization::LogTransform, None);

        assert!(expected_out_dense.abs_diff_eq(&norm_mat.to_dense(), 1e-6));
    }
}
