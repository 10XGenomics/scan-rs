#![cfg_attr(rustfmt, rustfmt_skip)]

use crate::betaln;
use approx::assert_abs_diff_eq;

const EPS: f64 = 1e-9;

#[test]
fn test_betaln() {
    // this corpus was generated using cargo-fuzzcheck, in an effort to exercise many codepaths
    assert_abs_diff_eq!(betaln(8.564_082_021_080_995e18, 1.386_093_852_236_636_4e19), -1.491_232_238_564_055_4e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(118.08777001498584, 29.265098832087116), -74.10555769218887, epsilon = EPS);
    assert!(betaln(f64::NAN, -1.7657446926169909).is_nan());
    assert_abs_diff_eq!(betaln(5.398338681547479, 2.147_500_404_792_697_2e154), -1914.5715726240255, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.385_202_117_257_239_1e19, 4.628_649_135_717_615e18), -1.040_163_797_913_003_6e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.638_010_105_015_565e18, 1.386_103_029_600_184_1e19), -1.041_719_188_571_934e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(9.036_329_806_426_149e18, 1.622_334_320_921_659_6e19), -1.647_196_556_970_098_7e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-1.7657446926169909, -162.80146510546749), 10.577476825297564, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.634_280_559_426_537e18, 1.383_400_306_316_712_3e19), -1.040_422_894_944_046_3e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-1.765744692616991, -1.7657446926169909), 3.458235649902897, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.2547013615612511e-118, 1.368259552498728e-45), 271.4781433879457, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.383_400_306_316_712_3e19, 4.629_000_981_586_115e18), -1.039_692_728_915_433_9e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(7.382743961174512, 1.843870169482912), -3.8434917074002493, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.616_673_275_268_028e18, 1.383_400_306_316_712_3e19), -1.037_986_048_328_033_9e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-28.265098832087116, -4.117071991915881), 13.154515266645687, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.618_907_516_987_492e18, 1.405_919_122_226_558e19), -1.044_737_603_647_347e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.844_618_167_351_181_3e19, 9.187_833_631_685_313e18), -1.757_310_497_792_589_8e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.383_400_306_316_712_3e19, 9.187_833_631_685_313e18), -1.548_545_050_409_002_6e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.835_819_185_410_804e18, 1.405_919_122_226_558e19), -1.074_668_290_327_065_4e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-7.07024205371492, -452.10876678681046), 37.75102087792084, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-452.10876678681046, -111.78355769512254), 279.2198448184731, epsilon = EPS);
    assert_abs_diff_eq!(betaln(5.799_787_067_520_067e18, 1.166_064_860_679_793_9e19), -1.109_965_005_302_405_5e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(6943589434778062.0, 1.689_249_881_133_755_3e157), -2.267_453_772_224_382_2e18, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-1.7657446926169909, -56.60588241278304), 8.197200479080747, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.737_085_585_928_223e18, 1.39601057769145e+19), -1.058_250_659_638_096_7e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-29629400.14014041, -1600644.767618594), 6314373.466539919, epsilon = EPS);
    assert_abs_diff_eq!(betaln(2.085_622_520_807_774_2e18, 9.291_141_891_761_947e18), -5.419_834_513_753_768e18, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-9.094204875685355e-97, -5.107525950735147e-195), 447.37337800523466, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.385_202_117_257_239_1e19, 4.628_930_610_694_325e18), -1.040_202_766_363_905_2e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.616_673_275_268_028e18, 1.405_919_122_226_558e19), -1.044_425_396_178_033e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(59.105897672459605, 43.23065543544351), -70.38530601066059, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-56.60588241278304, -56.60588241278304), 76.65483272577725, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.638_010_105_015_565e18, 1.386_093_852_236_636_4e19), -1.04171653963274e+19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-1.7657446926169909, -0.5693835497684786), 2.0781891908163246, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-474103171.8672542, -162.80146510546749), 2584.9772090911865, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-1.7657446926169909, -4.287_389_614_627_973e-5), 10.057158809367412, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.385_202_117_257_239_1e19, 6.915_735_149_216_515e18), -1.321_421_779_745_361e19, epsilon = EPS);
    assert!(betaln(-8.241712458692524, -162.80146510546749) == f64::INFINITY);
    assert_abs_diff_eq!(betaln(-28.265098832087116, -1.7657446926169909), 5.0054203106448965, epsilon = EPS);
    assert!(betaln(-2.322_612_297_625_279_4e253, -1.2715391977559994e-274) == f64::INFINITY);
    assert!(betaln(1.794_278_403_330_222_6e308, 1.557_376_483_595_712e306).is_nan());
    assert_abs_diff_eq!(betaln(-29629400.14014041, -111.78355769512254), 1505.4505025744438, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.384_301_844_015_179_6e19, 4.610_982_872_180_849_7e18), -1.037_456_687_286_203_2e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.634_280_559_426_537e18, 1.386_103_029_600_184_1e19), -1.041_203_118_368_725_4e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.103_094_102_965_431_3e19, 6.893_429_634_776_689e17), -2.621_802_305_324_318_7e18, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.377_656_445_890_065_6e19, 5.832_459_666_024_461e18), -1.193_557_110_803_542_8e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(7.382743961174512, 7.382743961174512), -9.951798388387317, epsilon = EPS);
    assert!(betaln(1.557_376_483_595_712e306, 1.557_376_483_595_712e306).is_nan());
    assert_abs_diff_eq!(betaln(1.384_420_132_655_426_8e19, 4.639_791_341_032_021e18), -1.041_479_419_444_291_2e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-32.980521918390856, -1990096217408107.8), 1080.0, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.623_415_377_287_433e18, 1.405_919_122_226_558e19), -1.045_367_273_420_868_8e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.481_164_943_573_686_3e18, 1.392_299_756_550_426_8e19), -1.02155160819499e+19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(-1.2125000044312759e-209, -4.287_389_614_627_973e-5), 481.0476000882714, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.383_852_367_111_193e19, 6.915_735_149_216_515e18), -1.320_874_951_715_874_4e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(6.915_735_149_216_515e18, 1.386_330_412_074_389_3e19), -1.321_878_552_855_838_7e19, epsilon = EPS);
    assert_abs_diff_eq!(betaln(7.382743961174512, 2.147_500_404_792_697_2e154), -2616.242597141291, epsilon = EPS);
    assert_abs_diff_eq!(betaln(1.560_609_709_012_785e77, 8.090_929_723_178_964e81), -1.850_261_536_401_283_8e78, epsilon = EPS);
    assert_abs_diff_eq!(betaln(4.618_907_516_987_492e18, 1.383_400_306_316_712_3e19), -1.038_295_545_603_175_2e19, epsilon = EPS);
    assert!(betaln(f64::NAN, f64::NAN).is_nan());
}
