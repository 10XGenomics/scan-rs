#![cfg_attr(rustfmt, rustfmt_skip)]

use crate::beta;
use approx::assert_abs_diff_eq;

const EPS: f64 = 1e-9;

#[test]
fn test_beta() {
    // this corpus was generated using cargo-fuzzcheck, in an effort to exercise many codepaths
    assert_abs_diff_eq!(beta(1.383_400_306_316_712_3e19, 4.617_764_007_446_047e18), 0.0, epsilon = EPS);
    assert!(beta(-1532161486053.388, -474103171.8672542) == f64::INFINITY);
    assert_abs_diff_eq!(beta(1.387_003_323_458_401e19, 6.915_735_149_216_515e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(4.638_010_105_015_565e18, 1.383_400_306_316_712_3e19), 0.0, epsilon = EPS);
    assert!(beta(f64::NAN, -113.08773949558939).is_nan());
    assert_abs_diff_eq!(beta(1.555_392_032_688_073_5e19, 2.440_645_976_483_144e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(-1.7657446926169909, -29629400.14014041), -30638352133959.63, epsilon = EPS);
    assert_abs_diff_eq!(beta(3.2241988673327137e-271, 1.0425232427011735e-274), 9.595_213_827_098_218e273, epsilon = EPS);
    assert_abs_diff_eq!(beta(-56.60588241278304, -113.08773949558939), -5.425_386_563_460_322e46, epsilon = EPS);
    assert!(beta(-162.80146510546749, -8.241712458692524) == f64::INFINITY);
    assert_abs_diff_eq!(beta(-1.765744692616991, -1.7657446926169909), 31.760889721983105, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.383_400_306_316_712_3e19, 4.629_000_981_586_115e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.843870169482912, 7.382743961174512), 0.021418682851062566, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.844_618_167_351_181_3e19, 9.187_833_631_685_313e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.096_282_851_175_119_5e19, 2.102_408_620_420_159e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(4.638_010_105_015_565e18, 1.386_110_066_689_363e19), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(-1.7657446926169909, -56.60588241278304), 3630.771645117701, epsilon = EPS);
    assert_abs_diff_eq!(beta(4.737_085_585_928_223e18, 1.39601057769145e+19), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(-4.4245749290999496e-71, -8.395865644994993e-169), -1.191_062_413_672_767e168, epsilon = EPS);
    assert!(beta(-2.359_141_922_873_785_7e-5, -2.592_645_972_592_420_4e228) == f64::INFINITY);
    assert_abs_diff_eq!(beta(1.843870169482912, 1.843870169482912), 0.21663412690198558, epsilon = EPS);
    assert_abs_diff_eq!(beta(59.105897672459605, 43.23065543544351), 2.704269640029204e-31, epsilon = EPS);
    assert_abs_diff_eq!(beta(-0.5693835497684786, -1.7657446926169909), 7.989987464004584, epsilon = EPS);
    assert_abs_diff_eq!(beta(-56.60588241278304, -56.60588241278304), 1.953_308_608_299_894_2e33, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.383_400_306_316_712_3e19, 2.016_717_332_773_783_3e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.385_202_117_257_239_1e19, 6.915_735_149_216_515e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.405_919_122_226_558e19, 4.835_819_185_410_804e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(-452.10876678681046, -7.07024205371492), -2.483_476_492_419_912e16, epsilon = EPS);
    assert_abs_diff_eq!(beta(-1.2178952235844431e-154, -32.980521918390856), -8.210_886_951_808_994e153, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.405_919_122_226_558e19, 4.610_982_872_180_849_7e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.387_003_323_458_401e19, 4.617_764_007_446_047e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(4.619_998_249_165_517e18, 1.383_400_306_316_712_3e19), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.401_247_057_456_911_8e19, 2.016_717_332_773_783_3e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.401_247_057_456_911_8e19, 6.915_735_149_216_515e18), 0.0, epsilon = EPS);
    assert!(beta(1.794_278_403_330_222_6e308, 1.557_376_483_595_712e306).is_nan());
    assert_abs_diff_eq!(beta(1.387_003_323_458_401e19, 4.618_907_516_987_492e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.361_516_317_242_946_8e18, 1.162_033_067_970_571_9e19), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(2.7031137979285784, 1.843870169482912), 0.11767363496816045, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.387_003_323_458_401e19, 4.623_415_377_287_433e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.405_919_122_226_558e19, 4.835_819_185_410_804e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(7.382743961174512, 7.382743961174512), 4.764_187_824_830_563e-5, epsilon = EPS);
    assert_abs_diff_eq!(beta(4.639_791_341_032_021e18, 1.384_420_132_655_426_8e19), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(118.08777001498584, 29.515106461936398), 4.3625356758147916e-33, epsilon = EPS);
    assert_abs_diff_eq!(beta(-452.10876678681046, -3.539042595484911), -1746178042.2987578, epsilon = EPS);
    assert!(beta(1.557_376_483_595_712e306, 1.557_376_483_595_712e306).is_nan());
    assert_abs_diff_eq!(beta(-452.10876678681046, -114.08777001498584), -2.996_001_208_666_052e123, epsilon = EPS);
    assert_abs_diff_eq!(beta(4.638_010_105_015_565e18, 1.386_138_214_187_034e19), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.382_667_580_748_518_8e19, 4.617_764_007_446_047e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(1.387_003_323_458_401e19, 4.619_998_249_165_517e18), 0.0, epsilon = EPS);
    assert!(beta(f64::NAN, -28.265098832087116).is_nan());
    assert!(beta(-29629400.14014041, -29629400.14014041) == f64::NEG_INFINITY);
    assert_abs_diff_eq!(beta(1.387_003_323_458_401e19, 4.616_673_275_268_028e18), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(beta(-452.10876678681046, -32.980521918390856), 6.539_829_143_139_209e52, epsilon = EPS);
    assert_abs_diff_eq!(beta(-162.80146510546749, -1.7657446926169909), 39240.97754407674, epsilon = EPS);
    assert!(beta(f64::NAN, f64::NAN).is_nan());
    assert_abs_diff_eq!(beta(-2.359_141_922_873_785_7e-5, -28.265098832087116), -42395.087027524765, epsilon = EPS);
    assert_abs_diff_eq!(beta(-1532161486053.388, -1.7657446926169909), 4.647_539_998_934_687e21, epsilon = EPS);
}
