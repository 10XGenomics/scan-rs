use anyhow::Error;
use ndarray::Axis;

/// Retrieve clustering information for analysis.hdf5 produced by xena pipeline
pub fn get_clustering(analysis_h5: &str, clustering_key: &str) -> Result<(u16, Vec<i16>), Error> {
    let clusterings = hdf5::File::open(analysis_h5)?
        .group("clustering")?
        .group(clustering_key)?;

    let clusters = clusterings
        .dataset("clusters")?
        .read_1d::<i64>()?
        .map(|&v| v as i16)
        .to_vec();

    let num_clusters = clusterings.dataset("num_clusters")?.read_scalar::<i64>()? as u16;

    Ok((num_clusters, clusters))
}

/// Read the result of differential expression computed on xena pipeline
pub fn get_differential_expression(analysis_h5: &str, clustering_key: &str) -> Result<Vec<Vec<f64>>, Error> {
    let mut ret = Vec::<Vec<f64>>::new();
    let differential_expression = hdf5::File::open(analysis_h5)?
        .group("all_differential_expression")?
        .group(clustering_key)?;

    differential_expression
        .dataset("data")?
        .read_2d::<f64>()?
        .axis_iter(Axis(0))
        .for_each(|v| ret.push(v.to_vec()));

    Ok(ret)
}

/// load clustering keys from analysis_h5
pub fn get_clustering_keys(analysis_h5: &str) -> Result<Vec<String>, Error> {
    let result = hdf5::File::open(analysis_h5)?.group("clustering")?.member_names()?;
    Ok(result)
}
