use sprs::CsMatI;
use sqz::AdaptiveMat;

use crate::label_class::LabelClass;

#[derive(Clone, Debug)]
pub struct GenericFeatureBarcodeMatrix<M> {
    pub name: String,
    pub barcodes: Vec<String>,
    pub feature_ids: Vec<String>,
    pub feature_names: Vec<String>,
    pub feature_types: LabelClass,
    pub matrix: M,
}

pub type LoupeMatrixType = CsMatI<f64, i64>;
pub type FeatureBarcodeMatrix = GenericFeatureBarcodeMatrix<LoupeMatrixType>;
pub type AdaptiveFeatureBarcodeMatrix = GenericFeatureBarcodeMatrix<AdaptiveMat>;

#[derive(Clone, Debug)]
pub struct MatrixMetadata {
    pub name: String,
    pub barcodes: Vec<String>,
    pub feature_ids: Vec<String>,
    pub feature_names: Vec<String>,
    pub feature_types: LabelClass,
    pub nnz: usize,
}

// FIXME -- temporary
impl PartialEq for FeatureBarcodeMatrix {
    fn eq(&self, other: &FeatureBarcodeMatrix) -> bool {
        self.name == other.name
    }
}

impl Eq for FeatureBarcodeMatrix {}
