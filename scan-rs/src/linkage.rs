use ndarray::{s, Array1, Array2, ArrayView1};
use noisy_float::types::n64;

fn euclidean(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    let n = x.dim();
    let mut d = 0.0;
    for i in 0..n {
        let k = x[i] - y[i];
        d += k * k;
    }
    d.sqrt()
}

fn pdist(x: &Array2<f64>) -> Array1<f64> {
    let (m, _) = x.dim();
    let mut d = Array1::<f64>::zeros((m * (m - 1) / 2,));
    let mut k = 0;
    for i in 0..m {
        for j in (i + 1)..m {
            d[k] = euclidean(x.slice(s![i, ..]), x.slice(s![j, ..]));
            k += 1;
        }
    }
    d
}

/// LinkageMethod is just Complete for now
pub trait LinkageMethod {
    /// The algorithm used by this LinkageMethod to hierarchically cluster
    fn algo(d: Array1<f64>, m: usize) -> Array2<f64>;
}

/// Complete linkage method corresponding to max(..)
pub struct Complete {}

impl LinkageMethod for Complete {
    fn algo(d: Array1<f64>, m: usize) -> Array2<f64> {
        nn_chain(d, m, f64::max)
    }
}

/// Hierarchically cluster with a LinkageMethod
pub fn linkage<T: LinkageMethod>(x: &Array2<f64>, _method: &T) -> Array2<f64> {
    let (m, _) = x.dim();
    let d = pdist(x);
    T::algo(d, m)
}

fn utidx(m: usize, a: usize, b: usize) -> usize {
    if a < b {
        m * a - (a * (a + 1) / 2) + b - a - 1
    } else {
        m * b - (b * (b + 1) / 2) + a - b - 1
    }
}

fn sort_by_column(x: Array2<f64>, col: usize) -> Array2<f64> {
    let mut d = x
        .slice(s![.., col])
        .iter()
        .enumerate()
        .map(|(i, &v)| (n64(v), i))
        .collect::<Vec<_>>();
    d.sort_unstable();
    let mut r = Array2::<f64>::zeros(x.dim());
    for (i, &(_, j)) in d.iter().enumerate() {
        r.slice_mut(s![i, ..]).assign(&x.slice(s![j, ..]));
    }
    r
}

fn nn_chain<F: Fn(f64, f64) -> f64>(mut d: Array1<f64>, m: usize, dist: F) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((m - 1, 4));

    let mut sizes = Array1::<usize>::ones((m,));
    let mut chain = Array1::<usize>::zeros((m,));
    let mut chain_length = 0;

    let mut a: usize;
    let mut b = 0;
    let mut curr_min: f64;

    for i in 0..(m - 1) {
        // rebuild the chain
        if chain_length == 0 {
            chain_length = 1;
            for j in 0..m {
                if sizes[j] > 0 {
                    chain[0] = j;
                    break;
                }
            }
        }

        loop {
            a = chain[chain_length - 1];

            if chain_length > 1 {
                b = chain[chain_length - 2];
                curr_min = d[utidx(m, a, b)];
            } else {
                curr_min = f64::INFINITY;
            }

            for c in 0..m {
                if sizes[c] == 0 || a == c {
                    continue;
                }

                let acdist = d[utidx(m, a, c)];
                if acdist < curr_min {
                    curr_min = acdist;
                    b = c;
                }
            }

            if chain_length > 1 && b == chain[chain_length - 2] {
                break;
            }

            chain[chain_length] = b;
            chain_length += 1;
        }

        // merge clusters a and b
        chain_length -= 2;

        if a > b {
            std::mem::swap(&mut a, &mut b);
        }

        let asz = sizes[a];
        let bsz = sizes[b];

        z[[i, 0]] = a as f64;
        z[[i, 1]] = b as f64;
        z[[i, 2]] = curr_min;
        z[[i, 3]] = (asz + bsz) as f64;

        sizes[a] = 0;
        sizes[b] = asz + bsz;

        for j in 0..m {
            let jsz = sizes[j];
            if jsz == 0 || j == b {
                continue;
            }
            let jadist = d[utidx(m, j, a)];
            let jbdist = d[utidx(m, j, b)];
            d[utidx(m, j, b)] = dist(jadist, jbdist);
        }
    }

    z = sort_by_column(z, 2);
    relabel(&mut z, m);

    z
}

fn relabel(z: &mut Array2<f64>, m: usize) {
    let mut uf = UnionFind::new(m);
    for i in 0..(m - 1) {
        let a = z[[i, 0]] as usize;
        let b = z[[i, 1]] as usize;
        let pa = uf.find(a);
        let pb = uf.find(b);
        if pa < pb {
            z[[i, 0]] = pa as f64;
            z[[i, 1]] = pb as f64;
        } else {
            z[[i, 0]] = pb as f64;
            z[[i, 1]] = pa as f64;
        }
        z[[i, 3]] = uf.merge(pa, pb) as f64;
    }
}

struct UnionFind {
    parents: Vec<usize>,
    sizes: Vec<usize>,
    next: usize,
}

impl UnionFind {
    fn new(m: usize) -> Self {
        UnionFind {
            parents: (0..(2 * m - 1)).collect::<Vec<_>>(),
            sizes: vec![1; 2 * m - 1],
            next: m,
        }
    }

    fn merge(&mut self, i: usize, j: usize) -> usize {
        self.parents[i] = self.next;
        self.parents[j] = self.next;
        let sz = self.sizes[i] + self.sizes[j];
        self.sizes[self.next] = sz;
        self.next += 1;
        sz
    }

    fn find(&mut self, mut i: usize) -> usize {
        let mut p = i;

        while self.parents[i] != i {
            i = self.parents[i];
        }

        while self.parents[p] != i {
            p = self.parents[p];
            self.parents[p] = i;
        }

        i
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    fn input_a() -> Array2<f64> {
        array![
            [0.61557404, 0.17137039],
            [0.6686267, 0.90885624],
            [0.26483002, 0.50614708],
            [0.49558047, 0.30861896],
            [0.38577965, 0.75407683],
            [0.3148579, 0.21179632],
            [0.89298659, 0.48151577],
            [0.22177291, 0.97322545],
            [0.00850986, 0.9995685],
            [0.98313583, 0.25529583]
        ]
    }

    fn input_b() -> Array2<f64> {
        array![
            [0.99702809, 0.93642583, 0.7998406],
            [0.7116703, 0.29029371, 0.6029036],
            [0.82347707, 0.73240751, 0.98303452],
            [0.59563889, 0.2280464, 0.6683355],
            [0.89014775, 0.60157901, 0.52721525],
            [0.29704329, 0.28184731, 0.36229336],
            [0.84291604, 0.30070089, 0.16268098],
            [0.12200112, 0.87085035, 0.41039911],
            [0.25650777, 0.40215799, 0.89952391],
            [0.3051844, 0.71880149, 0.71409149],
            [0.59377061, 0.41668407, 0.61616135]
        ]
    }

    #[test]
    fn test_pdist() {
        let a = input_a();
        let d = pdist(&a);
        let expected_d = array![
            0.73939161, 0.48486781, 0.18230641, 0.62638028, 0.30342124, 0.41611042, 0.89333693, 1.02685881, 0.37702142,
            0.57028627, 0.62468375, 0.32242692, 0.78169361, 0.48265643, 0.45146618, 0.66632047, 0.72529801, 0.30374846,
            0.27585862, 0.29857186, 0.6286393, 0.46905876, 0.55602583, 0.76084794, 0.45879073, 0.20502505, 0.43338774,
            0.71879925, 0.84536918, 0.49046263, 0.54689857, 0.5758024, 0.27372309, 0.45010961, 0.77821391, 0.63795091,
            0.76709787, 0.84524206, 0.66969217, 0.8320494, 1.02502571, 0.24352074, 0.21488389, 1.04646856, 1.22631056
        ];
        assert!(d.abs_diff_eq(&expected_d, 1e-7));

        let b = input_b();
        let d = pdist(&b);
        let expected_d = array![
            0.73328027, 0.32450492, 0.82474741, 0.44482623, 1.0535176, 0.91316433, 0.96001928, 0.91855823, 0.73031565,
            0.68299792, 0.59368753, 0.1470352, 0.36671714, 0.47945822, 0.45948853, 0.84959622, 0.55468023, 0.60100953,
            0.17335116, 0.63665206, 0.4788865, 0.9302992, 0.92721541, 0.91604926, 0.6614325, 0.58407428, 0.53576326,
            0.49616228, 0.4309474, 0.56754837, 0.83908334, 0.44583949, 0.57209963, 0.1957289, 0.69368617, 0.4750197,
            0.82231522, 0.76149991, 0.62517674, 0.36046769, 0.58153042, 0.61634284, 0.55202758, 0.56103222, 0.41313074,
            0.95192069, 0.94715665, 0.87636578, 0.53025464, 0.6906574, 0.38588086, 0.68641938, 0.37015903, 0.4407398,
            0.42912382
        ];
        assert!(d.abs_diff_eq(&expected_d, 1e-7));
    }

    #[test]
    fn test_linkage() {
        let a = input_a();
        let z = linkage(&a, &Complete {});
        let expected_z = array![
            [0., 3., 0.18230641, 2.],
            [7., 8., 0.21488389, 2.],
            [6., 9., 0.24352074, 2.],
            [2., 4., 0.27585862, 2.],
            [5., 10., 0.30342124, 3.],
            [11., 13., 0.55602583, 4.],
            [1., 15., 0.66632047, 5.],
            [12., 14., 0.66969217, 5.],
            [16., 17., 1.22631056, 10.],
        ];
        assert!(expected_z.abs_diff_eq(&z, 1e-7));

        let b = input_b();
        let z = linkage(&b, &Complete {});
        let expected_z = array![
            [1., 3., 0.1470352, 2.],
            [10., 11., 0.1957289, 3.],
            [0., 2., 0.32450492, 2.],
            [8., 9., 0.37015903, 2.],
            [4., 6., 0.4750197, 2.],
            [5., 12., 0.47945822, 4.],
            [14., 16., 0.60100953, 6.],
            [7., 17., 0.84959622, 7.],
            [13., 15., 0.92721541, 4.],
            [18., 19., 1.0535176, 11.]
        ];
        assert!(expected_z.abs_diff_eq(&z, 1e-7));
    }
}
