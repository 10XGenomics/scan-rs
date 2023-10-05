use anyhow::Error;
use ndarray::Array1;

/*
import (
    "cloupe/sparseio"
    "github.com/gonum/floats"
    "github.com/gonum/integrate/quad"
    "github.com/gonum/matrix/mat64"
    "github.com/gonum/stat"
    "github.com/gonum/stat/distuv"
    "math"
)

pub struct SseqParams {
    NumCells: usize,          // `json:"N"`
    NumGenes: usize,               //  `json:"G"`
    SizeFactors: Array1<f64>,   //`json:"size_factors"`
    GeneMeans: Array1<f64>, // `json:"mean_g"`
    GeneVariances: Array1<f64>, // `json:"var_g"`
    UseGenes: Vec<bool>, //      `json:"use_g"`
    GeneMomentPhi:Array1<f64>, // `json:"phi_mm_g"`
    ZetaHat: f64, //       `json:"zeta_hat"`
    Delta: f64, //       `json:"delta"`
    GenePhi: Array1<f64>, // `json:"phi_g"`
}


struct DiffExpResult {
    RowIndices:         Vec<usize>,  // `json:"rowIndices"`
    GenesTested:        Vec<bool>,   // `json:"tested"`
    SumsIn:             Array1<f64>, // `json:"sum_a"`
    SumsOut:            Array1<f64>, // `json:"sum_b"`
    CommonMeans:        Array1<f64>, // `json:"common_mean"`
    CommonDispersions:  Array1<f64>, // `json:"common_dispersion"`
    NormalizedMeansIn:  Array1<f64>, // `json:"norm_mean_a"`
    NormalizedMeansOut: Array1<f64>, // `json:"norm_mean_b"`
    PValues:            Array1<f64>, // `json:"p_values"`
    AdjustedPValues:    Array1<f64>, // `json:"adjusted_p_values"`
    Log2FoldChanges:    Array1<f64>, // `json:"log2_fold_change"`
}

struct SparseMatrixIO {}
impl SparseMatrixIO {
    fn rows(&self) -> usize { 0 }
    fn cols(&self) -> usize { 0 }
}
/**
 * Port of SingleGenomeAnalysis.compute_sseq_params
 *
 * Compute global parameters for the sSeq differential expression method.
 * The key parameters are the shrunken gene-wise dispersions.
 *
 * This method was published in:
 * Yu D, et al. (2013) Shrinkage estimation of dispersion in Negative Binomial models for RNA-seq experiments with small sample size.
 * Bioinformatics. 29: 1275-1282. doi: 10.1093/bioinformatics/btt143
 * Args:
 *   mtx - Sparse matrix (csc) of counts (gene x cell)
 *   zeta_quantile - Quantile of method-of-moments dispersion estimates to use as the shrinkage target zeta.
 *   umiCounts - precomputed UMI counts for all cells in the dataset (nullable)
 * Returns:
 * A struct containing the sSeq parameters and some diagnostic info.
 */
fn ComputeSseqParams(mtx: SparseMatrixIO, colIndices: Option<&Vec<usize>>, rowIndices: Option<&Vec<usize>>, zeta_quantile: f64,
                    umiCounts: &Vec<f64>) { // -> Result<SseqParams, Error> {
    let num_cells =
        if let Some(ci) = colIndices {
            ci.len()
        } else {
            mtx.cols()
        };

    let numGenes =
        if let Some(ri) = rowIndices {
            ri.len()
        } else {
            mtx.rows()
        };


    let sizeFactors: Array1<f64> = EstimateSizeFactors(mtx, colIndices, umiCounts)?;
    let sseqParams: SseqParams = mtx.GetSseqParamInfo(sizeFactors.RawVector().Data, colIndices, rowIndices)?;

    /*
    mean_g := mat64.NewVector(len(sseqParams.NormalizedGeneMeans), sseqParams.NormalizedGeneMeans)
    mean_sq_g := mat64.NewVector(numGenes, sseqParams.NormalizedSquaredGeneMeans)

    // this may be confusing...
    _geneMeansSquaredVec := mat64.NewVector(numGenes, nil)
    _geneMeansSquaredVec.MulElemVec(mean_g, mean_g)

    // TODO: to save memory, can reuse geneSquaredMeansVec here
    var_g := mat64.NewVector(numGenes, nil)
    var_g.SubVec(mean_sq_g, _geneMeansSquaredVec)
    */

    let gene_means_sq = sseqParams.NormalizedGeneMeans.map(|x| x*x);
    let var_g = sseqParams.NormalizedSquaredGeneMeans - gene_means_sq;

    // would like a nice way to subscript by condition but that doesn't appear to be in Vector support
    // for now, just do the calculation on all values and then wipe out the values for which variance is zero
    //_cellScaledVarianceVec := mat64.NewVector(numGenes, nil)
    //_cellScaledVarianceVec.ScaleVec(float64(numCells), var_g)

    let cell_scaled_var = var_g * (num_cells as f64);

    let reciprocalSum =
        if let Some(ci) = colIndices {
            let s = 0.0;
            for i in ci {
                s += 1.0 / sizeFactors[i];
            }
            s
        } else {
            sizeFactors.iter().map(|x| 1.0/x).sum()
        };

    //_scaledMeansVec := mat64.NewVector(numGenes, nil)
    //_scaledMeansVec.ScaleVec(reciprocalSum, mean_g)
    let scaled_means = sseqParams.NormalizedGeneMeans * reciprocalSum;


}
    /*
    _scaledMeansSquaredVec := mat64.NewVector(numGenes, nil)
    _scaledMeansSquaredVec.ScaleVec(reciprocalSum, _geneMeansSquaredVec)

    _numeratorVec := mat64.NewVector(numGenes, nil)
    _numeratorVec.SubVec(_cellScaledVarianceVec, _scaledMeansVec)

    phi_mm_g := mat64.NewVector(numGenes, nil)
    phi_mm_g.DivElemVec(_numeratorVec, _scaledMeansSquaredVec)

    _geneMomentPhiSlice := phi_mm_g.RawVector().Data
    _varianceSlice := var_g.RawVector().Data

    // count then allocate
    nzvNumElements := 0
    for _, variance := range var_g.RawVector().Data {
        if variance > 0 {
            nzvNumElements += 1
        }
    }
    varianceAboveZeroIndices := make([]int, nzvNumElements)
    currentVarianceIndex := 0
    for idx, variance := range _varianceSlice {
        if variance > 0 {
            varianceAboveZeroIndices[currentVarianceIndex] = idx
            currentVarianceIndex++
        }
    }

    // NOTE: this edits the backing slice in place
    for idx, val := range _geneMomentPhiSlice {
        if _varianceSlice[idx] <= 0 {
            _geneMomentPhiSlice[idx] = 0
        } else if val < 0 {
            _geneMomentPhiSlice[idx] = 0
        }
    }

    // oh this sucks-- again, no conditional support, so we must create
    // a sub index of phi_mm_g for which variance is nonzero
    nzvGeneMomentPhi := make([]float64, len(varianceAboveZeroIndices))
    for idx, varianceIndex := range varianceAboveZeroIndices {
        nzvGeneMomentPhi[idx] = _geneMomentPhiSlice[varianceIndex]
    }
    nzvGeneMomentPhiVec := mat64.NewVector(nzvNumElements, nzvGeneMomentPhi)

    // check for nans
    nonNaN := 0
    for _, val := range nzvGeneMomentPhiVec.RawVector().Data {
        if !math.IsNaN(val) {
            nonNaN += 1
        }
    }
    nonNanGeneMomentPhiVec := make([]float64, nonNaN)
    idx := 0
    for _, val := range nzvGeneMomentPhiVec.RawVector().Data {
        if !math.IsNaN(val) {
            nonNanGeneMomentPhiVec[idx] = val
            idx += 1
        }
    }

    // d13f539 fix from cellranger/diffexp.py
    useZeta := false
    for _, val := range nzvGeneMomentPhiVec.RawVector().Data {
        if val > 0 {
            useZeta = true
            break
        }
    }

    zetaHat := Quantile(nonNanGeneMomentPhiVec, zeta_quantile)

    nzvPhiMean := stat.Mean(nzvGeneMomentPhiVec.RawVector().Data, nil)
    _deltaNumVec := mat64.NewVector(nzvNumElements, nil)
    _deltaNumVec.SubVec(nzvGeneMomentPhiVec, VectorFull(nzvNumElements, nzvPhiMean))
    _deltaSquareNumVec := mat64.NewVector(nzvNumElements, nil)
    _deltaSquareNumVec.MulElemVec(_deltaNumVec, _deltaNumVec)
    _deltaDenomVec := mat64.NewVector(nzvNumElements, nil)
    _deltaDenomVec.SubVec(nzvGeneMomentPhiVec, VectorFull(nzvNumElements, zetaHat))
    _deltaSquareDenomVec := mat64.NewVector(nzvNumElements, nil)
    _deltaSquareDenomVec.MulElemVec(_deltaDenomVec, _deltaDenomVec)

    _deltaNum := VectorSum(_deltaSquareNumVec) / float64(numGenes-1)
    _deltaDenom := VectorSum(_deltaSquareDenomVec) / float64(numGenes-2)
    delta := _deltaNum / _deltaDenom

    phi_g := Full(numGenes, math.NaN())
    for varianceIdx, originalIdx := range varianceAboveZeroIndices {
        // d13f539 fix in cellranger/analysis/diffexp.py
        if useZeta {
            phi_g[originalIdx] = (1-delta)*nzvGeneMomentPhi[varianceIdx] + delta*zetaHat
        } else {
            phi_g[originalIdx] = 0.0
        }
    }

    useGenes := make([]bool, numGenes)
    for _, varianceIdx := range varianceAboveZeroIndices {
        useGenes[varianceIdx] = true
    }

    return &SseqParams{
        NumGenes:      numGenes,
        NumCells:      numCells,
        SizeFactors:   sizeFactors,
        GeneMeans:     mean_g,
        GeneVariances: var_g,
        UseGenes:      useGenes, // TODO: refer to index of array, or index of row if rowIndices != nil?
        GeneMomentPhi: phi_mm_g,
        ZetaHat:       zetaHat,
        Delta:         delta,
        GenePhi:       mat64.NewVector(len(phi_g), phi_g),
    }, nil
}

*/

/*
/**
 * Port of SingleGenomeAnalysis.estimate_size_factors
 *
 * Estimate size factors (related to cell RNA content and GEM-to-GEM technical variance)
 * Args:
 *   x - Sparse matrix (csc) of counts (gene x cell)
 *   colIndices - a subset of column indices to use for the calculation
 *   umiCounts - precomputed set of UMI counts per cell, nullable
 * Returns:
 *   Array of floats, one per cell, and an error if there is an issue.
 */

func EstimateSizeFactors(mtx *sparseio.SparseMatrixFloat64IO, colIndices []int64, umiCounts []float64) (*mat64.Vector, error) {
    countsPerCell := make([]float64, mtx.NumCols())
    if umiCounts != nil {
        copy(countsPerCell, umiCounts)
    } else {
        counts, err := mtx.GetColSums()
        if err != nil {
            return nil, err
        }
        countsPerCell = counts
    }
    if colIndices != nil {
        // zero out if not used
        filteredCountsPerCell := make([]float64, mtx.NumCols())
        for _, originalIndex := range colIndices {
            filteredCountsPerCell[originalIndex] = countsPerCell[originalIndex]
        }
        countsPerCell = filteredCountsPerCell
    }

    countsVec := mat64.NewVector(len(countsPerCell), countsPerCell)
    var medianValue float64
    if colIndices != nil {
        filteredValues := make([]float64, len(colIndices))
        for filteredIdx, cellIdx := range colIndices {
            filteredValues[filteredIdx] = countsPerCell[cellIdx]
        }
        medianValue = Median(filteredValues)
    } else {
        medianValue = Median(countsVec.RawVector().Data)
    }
    // since this is sparse we expect the median not to be zero
    countsVec.ScaleVec(1/medianValue, countsVec)
    return countsVec, nil
}

/**
 * Log(PMF) of negative binomial distribution with mean mu and dispersion phi,
 * conveniently parameterized.
 * Args:
 *  k (int) - NB random variable
 *  u (float) - mean
 *  phi (float) - dispersion
 * Returns:
 *  The log of the pmf at k.
 */
func NegativeBinomialLogPMF(k float64, mu float64, phi float64) float64 {
    r := 1.0 / phi
    lgammark, _ := math.Lgamma(r + k)
    lgammar, _ := math.Lgamma(r)
    lgammak1, _ := math.Lgamma(k + 1.0)
    return lgammark - (lgammar + lgammak1) + (float64(k) * math.Log(mu/(r+mu))) + (r * math.Log(r/(r+mu)))
}

/**
 * increasingInts is a fixed slice in memory from which to draw increasing arange slices.
 * decreasingInts is a fixed slice in memory from which to draw decreasing arange slices.
 */
func NegativeBinomialExactTests(countsIn []int32, countsOut []int32,
    sizeFactorIn float64, sizeFactorOut float64, mus []float64, phis []float64) []float64 {

    if len(countsIn) != len(countsOut) {
        panic("in/out lengths must be equal")
    }
    if len(countsIn) != len(mus) {
        panic("in/out/mu lengths must be equal")
    }
    if len(countsIn) != len(phis) {
        panic("in/out/mu/phi lengths must be equal")
    }

    maxSum := int32(0)
    for idx, _ := range countsIn {
        if countsIn[idx]+countsOut[idx] > maxSum {
            maxSum = countsIn[idx] + countsOut[idx]
        }
    }
    increasingInts := AInt32Range(0, maxSum+1)
    decreasingInts := AInt32Range(maxSum+1, -1)

    pValues := make([]float64, len(countsIn))

    var all_x_a []int32
    var all_x_b []int32
    for gidx := 0; gidx < len(countsIn); gidx++ {
        // start e7d0840 fix in cellranger/diffexp/analysis.py
        if sizeFactorIn == 0 || sizeFactorOut == 0 {
            pValues[gidx] = 1.0
            continue
        }
        // end e7d0840 fix in cellranger/diffexp/analysis.py
        countA := countsIn[gidx]
        countB := countsOut[gidx]
        if countsIn[gidx]+countsOut[gidx] == 0 {
            pValues[gidx] = 1.0
            continue
        }
        if phis[0] == 0 {
            pValues[gidx] = 1.0
            continue
        }

        all_x_a = increasingInts[:countA+countB+int32(1)]
        all_x_b = decreasingInts[int32(len(decreasingInts))-countB-countA-1:]

        log_p_obs := NegativeBinomialLogPMF(float64(countA), sizeFactorIn*mus[gidx], phis[gidx]/sizeFactorIn) +
            NegativeBinomialLogPMF(float64(countB), sizeFactorOut*mus[gidx], phis[gidx]/sizeFactorOut)

        log_p_all := make([]float64, len(all_x_a))
        for aidx, _ := range all_x_a {
            log_p_all[aidx] = NegativeBinomialLogPMF(float64(all_x_a[aidx]), sizeFactorIn*mus[gidx], phis[gidx]/sizeFactorIn) +
                NegativeBinomialLogPMF(float64(all_x_b[aidx]), sizeFactorOut*mus[gidx], phis[gidx]/sizeFactorOut)
        }
        moreExtreme := true
        lessExtreme := make([]float64, 0, len(log_p_all))
        for _, log_p := range log_p_all {
            // start d5b6308 fix in cellranger/diffexp/analysis.py (< vs <=)
            if log_p <= log_p_obs {
                // end d5b6308 fix in cellranger/diffexp/analysis.py
                moreExtreme = false
                lessExtreme = lessExtreme[:len(lessExtreme)+1]
                lessExtreme[len(lessExtreme)-1] = log_p
            }
        }
        if moreExtreme {
            pValues[gidx] = 0.0
            continue
        }
        pValues[gidx] = math.Exp(floats.LogSumExp(lessExtreme) - floats.LogSumExp(log_p_all))
    }
    return pValues
}

/**
 * Compute p-value for a pairwise exact test using a fast beta approximation
 * to the conditional joint distribution of (x_a, x_b).
 * Robinson MD and Smyth GK (2008). Small-sample estimation of negative binomial dispersion,
 *	with applications to SAGE data. Biostatistics, 9, 321-332
 *  "It is based a method-of-moments gamma approximation to the negative binomial distribution."
 *      - Personal communication w/ author
 */
func NegativeBinomialAsymptoteTests(countsIn []int32, countsOut []int32,
    sizeFactorA float64, sizeFactorB float64,
    mus []float64, phis []float64) []float64 {

    if len(countsIn) != len(countsOut) {
        panic("in/out lengths must be equal")
    }
    if len(countsIn) != len(mus) {
        panic("in/out/mu lengths must be equal")
    }
    if len(countsIn) != len(phis) {
        panic("in/out/mu/phi lengths must be equal")
    }

    alphas := make([]float64, len(countsIn))
    betas := make([]float64, len(countsOut))

    for idx, _ := range countsIn {
        alphas[idx] = sizeFactorA * mus[idx] / (1 + phis[idx]*mus[idx])
        betas[idx] = (sizeFactorB / sizeFactorA) * alphas[idx]
    }

    // this estimate is only really accurate when alpha > 1 and beta > 1, but since the median is just
    // used to figure out whether to compute the CDF from the left or right side for best integration accuracy,
    // it can be fairly off without much penalty.
    estimatedMedians := make([]float64, len(alphas))
    // https://en.wikipedia.org/wiki/Beta_distribution#Median
    for idx, _ := range alphas {
        estimatedMedians[idx] = (alphas[idx] - (1.0 / 3.0)) / (alphas[idx] + betas[idx] - (2.0 / 3.0))
    }
    totalCounts := make([]float64, len(countsIn))
    for idx, _ := range countsIn {
        totalCounts[idx] = float64(countsIn[idx] + countsOut[idx])
    }

    pValues := make([]float64, len(countsIn))
    for idx, _ := range countsIn {
        if (float64(countsIn[idx])+0.5)/totalCounts[idx] < estimatedMedians[idx] {
            pValues[idx] = 2 * BetaCDFLeft((float64(countsIn[idx])+0.5)/totalCounts[idx], alphas[idx], betas[idx])
        } else {
            pValues[idx] = 2 * BetaCDFRight((float64(countsIn[idx])-0.5)/totalCounts[idx], alphas[idx], betas[idx])
        }
    }
    return pValues
}

/**
 * Compute CDF to left of x
 */
func BetaCDFLeft(x float64, alpha float64, beta float64) float64 {
    betaDist := distuv.Beta{Alpha: alpha, Beta: beta}
    return quad.Fixed(betaDist.Prob, 0, x, 64, nil, 0)
}

/**
 * Compute CDF to right of x
 */
func BetaCDFRight(x float64, alpha float64, beta float64) float64 {
    betaDist := distuv.Beta{Alpha: alpha, Beta: beta}
    return quad.Fixed(betaDist.Prob, x, 1, 64, nil, 0)
}

func SseqDifferentialExpression(mtx *sparseio.SparseMatrixFloat64IO, rowIndices []int64, inColIndices []int64, outColIndices []int64,
    params *SseqParams, asymptoticCutoff int32) (*DiffExpResult, error) {

    var numGenes int64
    if rowIndices != nil {
        numGenes = int64(len(rowIndices))
    } else {
        numGenes = mtx.NumRows()
    }
    sizeFactorIn := float64(0.0)
    sizeFactorOut := float64(0.0)

    rawSizeFactors := params.SizeFactors.RawVector().Data
    for _, cellIdx := range inColIndices {
        sizeFactorIn += rawSizeFactors[cellIdx]
    }
    for _, cellIdx := range outColIndices {
        sizeFactorOut += rawSizeFactors[cellIdx]
    }

    geneSumsInF, err := mtx.GetFilteredRowSums(inColIndices, rowIndices)
    if err != nil {
        return nil, err
    }
    geneSumsOutF, err := mtx.GetFilteredRowSums(outColIndices, rowIndices)
    if err != nil {
        return nil, err
    }

    geneSumsIn := make([]int32, len(geneSumsInF))
    geneSumsOut := make([]int32, len(geneSumsOutF))
    for idx, inF := range geneSumsInF {
        geneSumsIn[idx] = RoundToInt32(inF)
    }
    for idx, outF := range geneSumsOutF {
        geneSumsOut[idx] = RoundToInt32(outF)
    }

    pValues := Full(int(numGenes), 1.0)
    bigCount := int64(0)
    for idx, _ := range geneSumsIn {
        if geneSumsIn[idx] > asymptoticCutoff && geneSumsOut[idx] > asymptoticCutoff {
            bigCount += 1
        }
    }

    // partitioning by condition seems pretty generic... is there a golangish way to do it?
    smallCount := numGenes - bigCount

    smallGeneSumsIn := make([]int32, smallCount)
    bigGeneSumsIn := make([]int32, bigCount)
    smallGeneSumsOut := make([]int32, smallCount)
    bigGeneSumsOut := make([]int32, bigCount)
    smallMus := make([]float64, smallCount)
    bigMus := make([]float64, bigCount)
    smallPhis := make([]float64, smallCount)
    bigPhis := make([]float64, bigCount)
    smallIndex := 0
    bigIndex := 0
    smallIndices := make([]int64, smallCount)
    bigIndices := make([]int64, bigCount)
    for idx, _ := range geneSumsIn {
        if geneSumsIn[idx] > asymptoticCutoff && geneSumsOut[idx] > asymptoticCutoff {
            bigGeneSumsIn[bigIndex] = geneSumsIn[idx]
            bigGeneSumsOut[bigIndex] = geneSumsOut[idx]
            bigMus[bigIndex] = params.GeneMeans.At(idx, 0)
            bigPhis[bigIndex] = params.GenePhi.At(idx, 0)
            bigIndices[bigIndex] = int64(idx)
            bigIndex += 1
        } else {
            smallGeneSumsIn[smallIndex] = geneSumsIn[idx]
            smallGeneSumsOut[smallIndex] = geneSumsOut[idx]
            smallMus[smallIndex] = params.GeneMeans.At(idx, 0)
            smallPhis[smallIndex] = params.GenePhi.At(idx, 0)
            smallIndices[smallIndex] = int64(idx)
            smallIndex += 1
        }
    }
    smallPValues := NegativeBinomialExactTests(smallGeneSumsIn, smallGeneSumsOut, sizeFactorIn, sizeFactorOut,
        smallMus, smallPhis)
    bigPValues := NegativeBinomialAsymptoteTests(bigGeneSumsIn, bigGeneSumsOut, sizeFactorIn, sizeFactorOut,
        bigMus, bigPhis)
    for idx, gidx := range smallIndices {
        pValues[gidx] = smallPValues[idx]
    }
    for idx, gidx := range bigIndices {
        pValues[gidx] = bigPValues[idx]
    }

    geneSumsInFVec := mat64.NewVector(len(geneSumsInF), geneSumsInF)
    geneSumsOutFVec := mat64.NewVector(len(geneSumsOutF), geneSumsOutF)
    normalizedMeansIn := mat64.NewVector(len(geneSumsInF), nil)
    normalizedMeansOut := mat64.NewVector(len(geneSumsOutF), nil)
    normalizedMeansIn.ScaleVec(1.0/sizeFactorIn, geneSumsInFVec)
    normalizedMeansOut.ScaleVec(1.0/sizeFactorOut, geneSumsOutFVec)

    log2InVec := VectorAddScalar(geneSumsInFVec, 1.0, false)
    log2InVec.ScaleVec(1.0/(1.0+sizeFactorIn), log2InVec)
    log2OutVec := VectorAddScalar(geneSumsOutFVec, 1.0, false)
    log2OutVec.ScaleVec(1.0/(1.0+sizeFactorOut), log2OutVec)
    log2InVec = VectorLog2(log2InVec, true)
    log2OutVec = VectorLog2(log2OutVec, true)
    log2ChangeVec := mat64.NewVector(len(geneSumsInF), nil)
    log2ChangeVec.SubVec(log2InVec, log2OutVec)

    return &DiffExpResult{
        RowIndices:         rowIndices,
        GenesTested:        params.UseGenes,
        SumsIn:             geneSumsInFVec,
        SumsOut:            geneSumsOutFVec,
        CommonMeans:        params.GeneMeans,
        CommonDispersions:  params.GenePhi,
        NormalizedMeansIn:  normalizedMeansIn,
        NormalizedMeansOut: normalizedMeansOut,
        PValues:            mat64.NewVector(len(pValues), pValues),
        AdjustedPValues:    mat64.NewVector(len(pValues), AdjustPValueBH(pValues)),
        Log2FoldChanges:    log2ChangeVec,
    }, nil
}

func AdjustPValueBH(pValues []float64) []float64 {
    indices := make([]int, len(pValues))
    adjPValues := make([]float64, len(pValues))
    copy(adjPValues, pValues)
    floats.Argsort(adjPValues, indices)
    indices = ReverseInt(indices)
    pValueVector := mat64.NewVector(len(pValues), ReverseFloat64(adjPValues))
    scaleVector := Reciprocate(mat64.NewVector(len(pValues), ARange(float64(len(pValues)), 0, -1)), float64(len(pValues)))
    scalePValueVector := mat64.NewVector(len(pValues), nil)
    scalePValueVector.MulElemVec(scaleVector, pValueVector)
    minimum := float64(1.0)
    minimums := make([]float64, len(pValues))
    for idx, val := range scalePValueVector.RawVector().Data {
        if minimum == float64(1.0) && val > float64(1.0) {
            minimums[idx] = minimum
        } else if val < minimum {
            minimum = val
            minimums[idx] = val
        } else {
            minimums[idx] = minimum
        }
    }

    for idx, oldIdx := range indices {
        adjPValues[oldIdx] = minimums[idx]
    }
    return adjPValues
}

*/
*/
