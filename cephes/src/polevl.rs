#[inline(always)]
pub(crate) fn polevl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().fold(0.0, |ans, &c| ans * x + c)
}

#[inline(always)]
pub(crate) fn p1evl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().fold(1.0, |ans, &c| ans * x + c)
}
