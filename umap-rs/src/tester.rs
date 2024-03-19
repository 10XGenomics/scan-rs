/// test mod
#[cfg(test)]
pub mod test {
    //use crate::test_data;
    use crate::dist::DistanceType;
    use crate::umap::Umap;
    #[cfg(feature = "plotly")]
    use crate::utils::plot_graph;
    use crate::utils::LabelledVector;
    use byteorder::{BigEndian, ReadBytesExt};
    use log::info;
    use ndarray::Array2;
    use std::env;
    use std::fs::File;
    use std::io::{Cursor, Read};
    use std::time::Instant;

    type Q = crate::utils::Q;

    fn setup(samples: usize) -> (Array2<Q>, Vec<LabelledVector>) {
        // Note: The MNIST data here consist of normalized vectors (so the CosineForNormalizedVectors distance function can be safely used)
        println!("cur_dir {}", env::current_dir().unwrap().display());
        let data_raw = match load_data("train") {
            Ok(data) => data,
            Err(err) => {
                println!("{err:?}");
                panic!("error");
            }
        };
        //let data_raw: Vec<LabelledVector> = data_raw.into_iter().filter(|q| q.id != "1").collect();
        let dim = data_raw[0].values.len();
        let data = data_raw
            .iter()
            .take(samples)
            .flat_map(|x| x.values.iter().map(|v| *v as Q).collect::<Vec<Q>>())
            .collect::<Vec<Q>>();

        let data = Array2::from_shape_vec((samples, dim), data).unwrap();
        (data, data_raw)
    }

    #[test]
    #[ignore] // needs test fixture
    fn completed_correlation() {
        let (data, data_raw) = setup(500);
        let distance_type = DistanceType::pearson();
        run_completed(&data, &data_raw, distance_type);
    }

    fn run_completed(data: &Array2<Q>, _data_raw: &[LabelledVector], distance_type: DistanceType) {
        let samples = data.shape()[0];
        let tick = Instant::now();
        #[cfg(feature = "plotly")]
        let colors_map: Vec<_> = _data_raw.iter().map(|item| item.id as i32).collect();
        let umap: Umap = Umap::new(Some(distance_type), 2, 0.01, 1.0, 10, None);

        info!("Initialize fit..");
        let mut state = umap.initialize_fit_parallelized(data, None, 1);
        info!(
            "Initialize Fit: for {} samples took {:.3}s",
            samples,
            tick.elapsed().as_millis() as f64 / 1000.0
        );

        info!("Calculating..");
        #[cfg(feature = "plotly")]
        let embedding = state.get_embedding();
        #[cfg(feature = "plotly")]
        plot_graph(true, embedding, &colors_map);

        let thread_pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        for i in 0..state.n_epochs {
            state.step(&thread_pool);
            if (i % 35) == 0 {
                info!("Completed {} of {}", i + 1, state.n_epochs);
                #[cfg(feature = "plotly")]
                let embedding = state.get_embedding();
                #[cfg(feature = "plotly")]
                plot_graph(true, embedding, &colors_map);
            }
        }
        #[cfg(feature = "plotly")]
        let embedding = state.get_embedding();
        #[cfg(feature = "plotly")]
        plot_graph(true, embedding, &colors_map);

        info!(
            "Done: for {} samples took {:.3}s",
            samples,
            tick.elapsed().as_millis() as f64 / 1000.0
        );
    }

    #[derive(Debug)]
    struct MnistData {
        sizes: Vec<i32>,
        data: Vec<u8>,
    }

    impl MnistData {
        fn new(f: &File) -> Result<MnistData, std::io::Error> {
            let mut gz = flate2::read::GzDecoder::new(f);
            let mut contents: Vec<u8> = Vec::new();
            gz.read_to_end(&mut contents)?;
            let mut r = Cursor::new(&contents);

            let magic_number = r.read_i32::<BigEndian>()?;

            let mut sizes: Vec<i32> = Vec::new();
            let mut data: Vec<u8> = Vec::new();

            match magic_number {
                2049 => {
                    sizes.push(r.read_i32::<BigEndian>()?);
                }
                2051 => {
                    sizes.push(r.read_i32::<BigEndian>()?);
                    sizes.push(r.read_i32::<BigEndian>()?);
                    sizes.push(r.read_i32::<BigEndian>()?);
                }
                _ => panic!(),
            }

            r.read_to_end(&mut data)?;

            Ok(MnistData { sizes, data })
        }
    }

    fn load_data(dataset_name: &str) -> Result<Vec<LabelledVector>, std::io::Error> {
        let cur_dir = env::current_dir()?;
        let filename = format!("testdata/{dataset_name}-labels-idx1-ubyte.gz");
        let filename = cur_dir.join(filename);

        println!("filename {} {}", filename.to_string_lossy(), filename.is_file());
        let label_data = &MnistData::new(&(File::open(filename.as_path()))?)?;

        let filename = format!("testdata/{dataset_name}-images-idx3-ubyte.gz");
        let filename = cur_dir.join(filename);
        let images_data = &MnistData::new(&(File::open(filename))?)?;

        let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

        let ret: Vec<LabelledVector> = label_data
            .data
            .iter()
            .enumerate()
            .map(|(i, &class)| {
                let start = i * image_shape;
                let image_data = &images_data.data[start..start + image_shape];
                let image_data: Vec<f32> = image_data.iter().map(|&x| x as f32 / 255.).collect();

                LabelledVector {
                    id: class,
                    values: image_data,
                }
            })
            .collect();

        Ok(ret)
    }
}
