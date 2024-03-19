use itertools::EitherOrBoth::Both;
use itertools::Itertools;
use ndarray::{arr1, Array1};
use num_traits::cast::{NumCast, ToPrimitive};
use num_traits::{Bounded, Zero};
use std::mem::size_of;
use std::rc::Rc;

/// Vector supporting abstracted storage of sparse data values.
/// The returned value can be materialized on-the-fly by the `get` method.
pub trait AbstractVec
where
    Self: std::marker::Sized,
{
    /// Output value type
    type Output: Zero + PartialEq;

    /// Internal type of the index into a non-zero value
    type Index: Clone;

    /// Length of the abstract vector
    fn len(&self) -> usize;

    /// True if the vector length is zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the value at position `i`. Should return `Self::Output::zero()`
    /// for 0 entrires.
    fn get(&self, i: usize) -> Self::Output;

    /// Get the `internal` index of positions `i`. Returns None
    /// if this position is beyond the last stored value
    fn get_index(&self, i: usize) -> Option<Self::Index>;

    /// Number of non-zero entries. Note this must be accurate, even for
    /// dense representations that might store some zeros.
    fn nnz(&self) -> usize;

    /// Get the ith non-zero entry, return a tuple of the position of the value in the full vector, and the value.
    fn get_nonzero(&self, i: Self::Index) -> (usize, Self::Output);

    /// Increment the given index value. This may require looking at the stored data.
    /// Returns None if no more indexes exist.
    fn incr(&self, index: Self::Index) -> Option<Self::Index>;

    /// Report number of heap bytes consumed by this vector
    fn mem_size(&self) -> usize;

    /// Return an iterator over the non-zero values in the vector
    fn iter(&self) -> AbsIter<Self> {
        AbsIter {
            idx: self.get_index(0),
            vec: self,
            end: self.len(),
        }
    }

    /// Iterate over the non-zero values in the given `range` of indices.
    fn iter_range(&self, range: std::ops::Range<usize>) -> AbsIter<Self> {
        AbsIter {
            idx: self.get_index(range.start),
            vec: self,
            end: range.end,
        }
    }
}

/// Trait for constructing a sparse vector
trait MemConstruct {
    type Output;

    /// Estimate the size in bytes to store a vector of length `len`
    /// and the given values in an value of this type.
    fn estimate_size(len: usize, values: &[Self::Output]) -> usize;

    /// Construct a sparse vector of length `len`, with non-zero entries
    /// defined by parallel array `values` and `indexes`. If `indexes == None`
    /// the data in values is treated as dense and `len == values.len()` must hold.
    fn construct(len: usize, values: &[Self::Output], indexes: Option<&[u32]>) -> Self;
}

/// Iterator over the non-zero values of an `AbstractVec`
#[derive(Debug)]
pub struct AbsIter<'s, T>
where
    T: AbstractVec,
{
    idx: Option<T::Index>,
    vec: &'s T,
    end: usize,
}

impl<'s, T: AbstractVec> Iterator for AbsIter<'s, T> {
    type Item = (usize, T::Output);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.idx.as_ref()?;

            let idx = self.idx.as_ref().unwrap().clone();
            let (pos, value) = self.vec.get_nonzero(idx.clone());

            if pos >= self.end {
                return None;
            }

            self.idx = self.vec.incr(idx);

            if value != T::Output::zero() {
                return Some((pos, value));
            }
        }
    }
}

/// Traditional sparse vector representatopn supporting lengths up to u32::MAX
/// Used as a fallback store for narrow bit-width representations.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SimpleSparse<T> {
    len: usize,
    indexes: Vec<u32>,
    values: Vec<T>,
}

impl<T: Clone + Zero + PartialEq> AbstractVec for SimpleSparse<T> {
    type Output = T;
    type Index = usize;

    fn len(&self) -> usize {
        self.len
    }

    fn nnz(&self) -> usize {
        self.indexes.len()
    }

    fn get(&self, i: usize) -> Self::Output {
        debug_assert!(i < self.len);

        match self.indexes.binary_search(&(i as u32)) {
            Ok(v) => self.values[v].clone(),
            Err(_) => T::zero(),
        }
    }

    fn get_index(&self, i: usize) -> Option<Self::Index> {
        debug_assert!(i < self.len);

        let idx = match self.indexes.binary_search(&(i as u32)) {
            Ok(v) => v,
            Err(v) => v,
        };

        if idx >= self.values.len() {
            None
        } else {
            Some(idx)
        }
    }

    fn get_nonzero(&self, i: usize) -> (usize, Self::Output) {
        (self.indexes[i] as usize, self.values[i].clone())
    }

    #[inline]
    fn incr(&self, index: usize) -> Option<usize> {
        let n = index + 1;
        if n < self.indexes.len() {
            Some(n)
        } else {
            None
        }
    }

    fn mem_size(&self) -> usize {
        self.nnz() * (size_of::<T>() + 4)
    }
}

impl<T: Clone + Zero> MemConstruct for SimpleSparse<T> {
    type Output = T;

    fn estimate_size(_len: usize, values: &[Self::Output]) -> usize {
        values.len() * (size_of::<T>() + size_of::<u32>())
    }

    fn construct(len: usize, values: &[Self::Output], indexes: Option<&[u32]>) -> Self {
        if indexes.is_none() {
            assert_eq!(
                len,
                values.len(),
                "must supply a value for each position when indexes == None"
            );
        }

        let v: Vec<u32>;
        let indexes = match indexes {
            Some(v) => v,
            None => {
                v = (0..(len as u32)).collect::<Vec<_>>();
                &v
            }
        };

        SimpleSparse {
            len,
            values: Vec::from(values),
            indexes: Vec::from(indexes),
        }
    }
}

/// Sparse vector with index compression. Index space is partitioned into 256 element
/// blocks. The start position and length of each block of occupied indexes are stored
/// in `block_starts` and `block_lengths`, and the offsets of occupied values within
/// the block are stored in `index_bytes`.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct CompressedIndexSparse<T> {
    len: usize,
    dense_data: T,
    index_bytes: Vec<u8>,
    block_starts: Vec<u32>,
}

impl<O: Zero + PartialEq + Eq, T: AbstractVec<Output = O>> AbstractVec for CompressedIndexSparse<T> {
    type Output = O;
    type Index = (usize, usize);

    fn len(&self) -> usize {
        self.len
    }

    fn nnz(&self) -> usize {
        self.index_bytes.len()
    }

    #[inline]
    fn get(&self, i: usize) -> Self::Output {
        let block = i >> 8;
        let offset = (i & 0xFF) as u8;

        let block_start = self.block_starts[block] as usize;
        let block_end = self.block_starts[block + 1] as usize;

        let index_slc = &self.index_bytes[block_start..block_end];

        match index_slc.binary_search(&offset) {
            Ok(v) => self.dense_data.get(v + block_start),
            Err(_) => O::zero(),
        }
    }

    /// Get the index of non-zero value at the smallest position greater or equal to `i`.
    /// Return None if this index doesn't exist.
    fn get_index(&self, i: usize) -> Option<Self::Index> {
        let block = i >> 8;
        let offset = (i & 0xFF) as u8;

        let block_start = self.block_starts[block] as usize;
        let block_end = self.block_starts[block + 1] as usize;

        let index_slc = &self.index_bytes[block_start..block_end];

        match index_slc.binary_search(&offset) {
            Ok(v) => {
                let main_idx = v + block_start;
                Some((block, main_idx))
            }
            // If don't find a value in this block, fast-forward to the start of the next block
            Err(v) => {
                let main_idx = v + block_start;
                let mut block_idx = block;

                if main_idx < self.index_bytes.len() {
                    while !(self.block_starts[block_idx] <= main_idx as u32
                        && self.block_starts[block_idx + 1] > main_idx as u32)
                    {
                        block_idx += 1
                    }
                    Some((block_idx, main_idx))
                } else {
                    None
                }
            }
        }
    }

    fn get_nonzero(&self, idx: Self::Index) -> (usize, O) {
        let (block_idx, main_idx) = idx;

        let pos = (block_idx << 8) | (self.index_bytes[main_idx] as usize);
        let value = self.dense_data.get(main_idx);
        (pos, value)
    }

    #[inline]
    fn incr(&self, idx: Self::Index) -> Option<Self::Index> {
        let (mut block_idx, mut main_idx) = idx;

        main_idx += 1;

        // Fast-forward the block counter to the block that contains this inde
        if main_idx < self.index_bytes.len() {
            while !(self.block_starts[block_idx] <= main_idx as u32
                && self.block_starts[block_idx + 1] > main_idx as u32)
            {
                block_idx += 1
            }
            Some((block_idx, main_idx))
        } else {
            None
        }
    }

    fn mem_size(&self) -> usize {
        self.dense_data.mem_size() + self.index_bytes.capacity() + 4 * self.block_starts.capacity()
    }
}

impl<O: Zero, T: MemConstruct<Output = O>> MemConstruct for CompressedIndexSparse<T> {
    type Output = O;

    fn estimate_size(len: usize, values: &[Self::Output]) -> usize {
        // Store:
        // - A dense array of length values.len() in T
        // - 1 byte per value for index within the block of 256
        // - 4 bytes for each block of 256, for the start position of the block
        T::estimate_size(values.len(), values) + values.len() + (len / 256) * size_of::<u32>()
    }

    fn construct(len: usize, values: &[Self::Output], indexes: Option<&[u32]>) -> Self {
        if indexes.is_none() {
            assert_eq!(
                len,
                values.len(),
                "must supply a value for each position when indexes == None"
            );
        }
        let dense = T::construct(values.len(), values, None);

        let total_blocks = round_up(len, 256) / 256;

        let mut index_bytes = Vec::new();
        let mut block_starts: Vec<u32> = Vec::new();

        let mut cur_block = 0;
        let mut cur_block_start = 0;

        let v: Vec<u32>;
        let indexes = match indexes {
            Some(v) => v,
            None => {
                v = (0..(len as u32)).collect::<Vec<_>>();
                &v
            }
        };

        for i in indexes {
            let block = i / 256;
            let offset = (i % 256) as u8;

            // flush last block
            if block > cur_block {
                block_starts.push(cur_block_start);
                cur_block = block;
                cur_block_start = index_bytes.len() as u32;
            }

            while block_starts.len() < block as usize {
                block_starts.push(index_bytes.len() as u32);
            }

            index_bytes.push(offset);
        }

        // flush last block
        block_starts.push(cur_block_start);

        // pad in empty blocks
        while block_starts.len() < total_blocks {
            block_starts.push(index_bytes.len() as u32);
        }

        // tack on extra value to give end of last block
        block_starts.push(index_bytes.len() as u32);

        CompressedIndexSparse {
            len,
            dense_data: dense,
            index_bytes,
            block_starts,
        }
    }
}

fn round_up(v: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return v;
    }

    let remainder = v % multiple;
    if remainder == 0 {
        v
    } else {
        v + multiple - remainder
    }
}

#[derive(Debug, Clone)]
struct IndexPattern {
    positions: Vec<u32>,
}

impl IndexPattern {
    fn get_index(&self, i: usize) -> Option<usize> {
        debug_assert!(i < self.positions.len());

        let idx = match self.positions.binary_search(&(i as u32)) {
            Ok(v) => v,
            Err(v) => v,
        };

        if idx >= self.positions.len() {
            None
        } else {
            Some(idx)
        }
    }

    fn get_pos(&self, idx: usize) -> usize {
        self.positions[idx] as usize
    }

    #[inline]
    fn incr(&self, index: usize) -> Option<usize> {
        let n = index + 1;
        if n < self.positions.len() {
            Some(n)
        } else {
            None
        }
    }
}

/// Struct for tracking the iteration over a PatternHybrid sparse vector
#[derive(Clone)]
pub struct HybridIndex {
    index4: Option<usize>,
    index8: Option<usize>,
    index_sp: Option<(usize, usize)>,
}

impl HybridIndex {
    fn is_none(&self) -> bool {
        self.index4.is_none() && self.index8.is_none() && self.index_sp.is_none()
    }
}

/// A sparse vector that uses shared 'pattern' vectors to indicate the positions of values that are likely to be non-zero,
/// and stores those values in a dense matrix, and stores remaining values in a normal `CompressedIndexSparse` vector.
/// The pattern vectors should be shared by many instances of `PatternHybrid` to amortize their memory consumption.
#[derive(Debug, Clone)]
pub struct PatternHybrid<TO: PartialOrd + Clone + ToPrimitive + Zero + PartialEq> {
    len: usize,

    nnz: usize,

    /// Indexes of values stored in dense_data_4
    pattern4: Rc<IndexPattern>,

    /// Values stored in 4-bit dense vector, corresponding to indexes in pattern4.
    dense_data_4: Dense4<TO>,

    /// Indexes of values stored in dense_data_4
    pattern8: Rc<IndexPattern>,

    /// Values stored in 4-bit dense vector, corresponding to indexes in pattern4.
    dense_data_8: DenseW<u8, TO>,

    /// Non-zero values not in the pattern vectors are stored here
    sparse_data: CompressedIndexSparse<Dense4<TO>>,
}

impl<TO: PartialOrd + Clone + ToPrimitive + Zero + PartialEq> PatternHybrid<TO> {
    /// Count the occupancy of features
    pub fn count_occupancy(len: usize, sample_values: &[(&[usize], &[u32])]) {
        //(IndexPattern, IndexPattern) {

        let mut count_4bits = vec![0; len];
        let mut count_8bits = vec![0; len];

        for (idxs, vals) in sample_values.iter().cloned() {
            for (idx, val) in idxs.iter().cloned().zip(vals) {
                if *val >= 15 {
                    count_8bits[idx] += 1;
                } else {
                    count_4bits[idx] += 1;
                }
            }
        }
    }

    /// Decide what features are stored in dense matrices
    pub fn choose_patterns(frac_small: Vec<f64>, frac_large: Vec<f64>) -> (Vec<usize>, Vec<usize>) {
        let mut pattern_4 = Vec::new();
        let mut pattern_8 = Vec::new();

        let mut pat4_count = 0;
        let mut pat8_count = 0;
        let mut sparse_count = 0;

        for i in 0..frac_small.len() {
            // Store in 4-bit sparse matrix w/ 32bit fallback
            let size_sparse = 1.5 * frac_small[i] + 8.0 * frac_large[i];

            // Store in 4bit dense matrix
            let size_4 = 0.5 + 8.0 * frac_large[i];

            // Store in a 1-byte dense_matrix
            let size_8 = 1.0;

            if size_4 < size_8 && size_4 < size_sparse {
                pattern_4.push(i);
                pat4_count += 1;
            } else if size_8 < size_4 && size_8 < size_sparse {
                pattern_8.push(i);
                pat8_count += 1;
            } else {
                sparse_count += 1;
            }
        }

        println!(
            "total genes: {}, sparse: {}, 4-bit dense: {}, 8-bit dense: {}",
            frac_small.len(),
            pat4_count,
            pat8_count,
            sparse_count
        );
        (pattern_4, pattern_8)
    }
}

impl<TO> AbstractVec for PatternHybrid<TO>
where
    TO: PartialOrd + Clone + ToPrimitive + Zero + PartialEq + From<u8> + Eq,
{
    type Output = TO;
    type Index = HybridIndex;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, i: usize) -> TO {
        if let Some(p) = self.pattern4.get_index(i) {
            return self.dense_data_4.get(p);
        }

        if let Some(p) = self.pattern8.get_index(i) {
            return self.dense_data_8.get(p);
        }

        self.sparse_data.get(i)
    }

    fn get_index(&self, i: usize) -> Option<HybridIndex> {
        if i < self.len {
            let index4 = self.pattern4.get_index(i);
            let index8 = self.pattern8.get_index(i);
            let index_sp = self.sparse_data.get_index(i);

            let idx = HybridIndex {
                index4,
                index8,
                index_sp,
            };

            if idx.is_none() {
                None
            } else {
                Some(idx)
            }
        } else {
            None
        }
    }

    fn nnz(&self) -> usize {
        self.nnz
    }

    #[inline]
    fn incr(&self, mut index: HybridIndex) -> Option<HybridIndex> {
        // determine which of the indexes contains the next non-zero value, and increment that index.
        let pos4 = index.index4.map(|v| self.pattern4.get_pos(v));
        let pos8 = index.index4.map(|v| self.pattern8.get_pos(v));
        let pos_sp = index.index_sp.map(|v| self.sparse_data.get_nonzero(v).0);

        let ipos4 = pos4.unwrap_or(usize::max_value());
        let ipos8 = pos4.unwrap_or(usize::max_value());
        let ipos_sp = pos4.unwrap_or(usize::max_value());

        if pos4.is_some() && ipos4 < ipos8 && ipos4 < ipos_sp {
            index.index4 = self.pattern4.incr(index.index4.unwrap());
        } else if pos8.is_some() && ipos8 < ipos4 && ipos8 < ipos_sp {
            index.index8 = self.pattern8.incr(index.index8.unwrap());
        } else if pos_sp.is_some() && ipos_sp < ipos4 && ipos_sp < ipos8 {
            index.index_sp = self.sparse_data.incr(index.index_sp.unwrap());
        } else {
            panic!("invalid state of HybridIndex for PatternHybrid sparse vector")
        }

        // check if we've gotten to the end of all the indexes
        if index.is_none() {
            return None;
        }

        Some(index)
    }

    fn get_nonzero(&self, index: Self::Index) -> (usize, TO) {
        // determine which of the indexes contains the next non-zero value, and increment that index.
        let pos4 = index.index4.map(|v| self.pattern4.get_pos(v));
        let pos8 = index.index4.map(|v| self.pattern8.get_pos(v));
        let pos_sp = index.index_sp.map(|v| self.sparse_data.get_nonzero(v).0);

        let ipos4 = pos4.unwrap_or(usize::max_value());
        let ipos8 = pos8.unwrap_or(usize::max_value());
        let ipos_sp = pos_sp.unwrap_or(usize::max_value());

        if ipos4 < ipos8 && ipos4 < ipos_sp {
            let v = self.dense_data_4.get(index.index4.unwrap());
            (ipos4, v)
        } else if ipos8 < ipos4 && ipos8 < ipos_sp {
            let v = self.dense_data_8.get(index.index8.unwrap());
            (ipos8, v)
        } else if ipos_sp < ipos4 && ipos_sp < ipos8 {
            self.sparse_data.get_nonzero(index.index_sp.unwrap())
        } else {
            //assert!(false, "invalid state of HybridIndex for PatternHybrid sparse vector");
            unreachable!("invalid state of HybridIndex for PatternHybrid sparse vector")
        }
    }

    fn mem_size(&self) -> usize {
        self.dense_data_4.mem_size() + self.dense_data_8.mem_size() + self.sparse_data.mem_size()
    }
}

/// Stores a vector of integer of type `TO`. The representation is a dense array of values of `TS` which
/// should have a narrower bit-width the `TO`. Values `>= TS::MAX` are encoded as `TS::MAX`, with a fallback
/// to a sparse array of `TO` values.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct DenseW<TS, TO> {
    nnz: usize,
    data: Vec<TS>,
    fallback: SimpleSparse<TO>,
}

impl<TS, TO> AbstractVec for DenseW<TS, TO>
where
    TS: Bounded + Into<TO> + NumCast + PartialEq + Copy,
    TO: PartialOrd + Clone + ToPrimitive + Zero + PartialEq,
{
    type Output = TO;
    type Index = usize;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, i: usize) -> TO {
        let v = self.data[i];
        if v == TS::max_value() {
            self.fallback.get(i)
        } else {
            v.into()
        }
    }

    fn get_index(&self, i: usize) -> Option<usize> {
        if i < self.len() {
            Some(i)
        } else {
            None
        }
    }

    fn nnz(&self) -> usize {
        self.nnz
    }

    #[inline]
    fn incr(&self, index: usize) -> Option<usize> {
        let n = index + 1;
        if n >= self.len() {
            None
        } else {
            Some(n)
        }
    }

    fn get_nonzero(&self, i: Self::Index) -> (usize, TO) {
        (i, self.get(i))
    }

    fn mem_size(&self) -> usize {
        self.data.capacity() * size_of::<TS>() + self.fallback.mem_size()
    }
}

impl<TS: Bounded + Into<TO> + NumCast + Copy, TO: PartialOrd + Clone + ToPrimitive + Zero> MemConstruct
    for DenseW<TS, TO>
{
    type Output = TO;

    fn estimate_size(len: usize, values: &[Self::Output]) -> usize {
        let thresh: TO = TS::max_value().into();
        let oversize = values.iter().filter(|x| **x >= thresh).count();
        let fallback_size = oversize * (size_of::<TO>() + size_of::<u32>());
        len * size_of::<TS>() + fallback_size
    }

    fn construct(len: usize, values: &[Self::Output], indexes: Option<&[u32]>) -> Self {
        let mut data: Vec<TS> = vec![NumCast::from(TO::zero()).unwrap(); len];
        let mut overflow_idxs: Vec<u32> = Vec::new();
        let mut overflow_vals: Vec<TO> = Vec::new();

        let thresh: TO = TS::max_value().into();

        for (i, v) in values.iter().cloned().enumerate() {
            let idx = indexes.map_or(i as u32, |idxs| idxs[i]);
            if v >= thresh {
                overflow_idxs.push(idx);
                overflow_vals.push(v);
                data[idx as usize] = TS::max_value();
            } else {
                data[idx as usize] = NumCast::from(v).unwrap();
            }
        }

        let fallback = SimpleSparse::construct(len, &overflow_vals, Some(&overflow_idxs));

        DenseW {
            nnz: values.len(),
            data,
            fallback,
        }
    }
}

/// Stores a vector of u32 values. The representation is a dense array of 4-bit byte-packed values.
/// Values `>= 15` are encoded as `15` in the 4-bit array, with actual value stored in sparse array of u32 values.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Dense4<T> {
    nnz: usize,
    len: usize,
    data: Vec<u8>,
    fallback: SimpleSparse<T>,
}

impl<T> AbstractVec for Dense4<T>
where
    T: PartialOrd + Clone + ToPrimitive + Zero + PartialEq + From<u8>,
{
    type Output = T;
    type Index = usize;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, i: usize) -> T {
        let slot = i >> 1;
        let which = i - (slot << 1);
        let v = self.data[slot];

        let raw_v = if which == 0 { v & 0xF } else { v >> 4 };

        if raw_v == 15 {
            AbstractVec::get(&self.fallback, i)
        } else {
            T::from(raw_v)
        }
    }

    fn get_index(&self, i: usize) -> Option<usize> {
        if i < self.len() {
            Some(i)
        } else {
            None
        }
    }

    fn nnz(&self) -> usize {
        self.nnz
    }

    #[inline]
    fn incr(&self, index: usize) -> Option<usize> {
        let n = index + 1;
        if n >= self.len() {
            None
        } else {
            Some(n)
        }
    }

    fn get_nonzero(&self, i: Self::Index) -> (usize, T) {
        (i, self.get(i))
    }

    fn mem_size(&self) -> usize {
        self.data.capacity() + self.fallback.mem_size()
    }
}

impl<T> MemConstruct for Dense4<T>
where
    T: PartialOrd + Clone + ToPrimitive + Zero + PartialEq + From<u8> + NumCast,
{
    type Output = T;

    fn estimate_size(len: usize, values: &[Self::Output]) -> usize {
        let thresh: T = <T as NumCast>::from(15u8).unwrap();
        let oversize = values.iter().filter(|x| **x >= thresh).count();
        let fallback_size = oversize * (size_of::<u32>() + size_of::<u32>());
        len / 2 + fallback_size
    }

    fn construct(len: usize, values: &[Self::Output], indexes: Option<&[u32]>) -> Self {
        if indexes.is_none() {
            assert_eq!(
                len,
                values.len(),
                "must supply a value for each position when indexes == None"
            );
        }

        let mut data: Vec<u8> = vec![0; len / 2 + 1];
        let mut overflow_idxs: Vec<u32> = Vec::new();
        let mut overflow_vals: Vec<T> = Vec::new();

        let thresh: T = <T as NumCast>::from(15u8).unwrap();

        let mut set_pos = |pos: usize, value: u8| {
            let slot = pos >> 1;
            let offset = pos & 0x1;

            let orig = data[slot];

            let res = if offset == 0 {
                (orig & 0xF0) | (value & 0x0F)
            } else {
                (orig & 0x0F) | ((value & 0x0F) << 4)
            };

            data[slot] = res;
        };

        for (i, v) in values.iter().cloned().enumerate() {
            let idx = indexes.map_or(i as u32, |idxs| idxs[i]);

            if v >= thresh {
                overflow_idxs.push(idx);
                overflow_vals.push(v);
                set_pos(idx as usize, 15);
            } else {
                set_pos(idx as usize, v.to_u8().unwrap());
            }
        }

        let fallback = SimpleSparse::construct(len, &overflow_vals, Some(&overflow_idxs));

        Dense4 {
            nnz: values.len(),
            len,
            data,
            fallback,
        }
    }
}

/// Sparse Vector encoding `u32` values, with length up to `u32::MAX`.
/// Adaptively uses the most memory-efficient storage format based on
/// the data being stored.
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum AdaptiveVec {
    /// Dense storage of 4-bit values, with fallback to a sparse array of u32 values
    D4(Dense4<u32>),

    /// Dense storage of 8-bit values, with fallback to a sparse array of u32 values
    D8(DenseW<u8, u32>),

    /// Dense storage of 16-bit values with fallback to a sparse array of u32 values
    D16(DenseW<u16, u32>),

    /// Compressed-index sparse storage, with values store as 4-bit dense array with u32 fallback.
    S4(CompressedIndexSparse<Dense4<u32>>),

    /// Compressed-index sparse storage, with values store as 4-bit dense array with u32 fallback.
    S8(CompressedIndexSparse<DenseW<u8, u32>>),
}

#[macro_export]
/// Execute an expression over all variants of an `AdaptiveVec`. Expression `e` should treat the
/// value `v` as an `impl AdaptiveVec`.
macro_rules! ada_expand {
    ($input:ident, $v:ident, $e:expr) => {{
        use $crate::vec::AdaptiveVec::*;
        match $input {
            D4($v) => $e,
            D8($v) => $e,
            D16($v) => $e,
            S4($v) => $e,
            S8($v) => $e,
        }
    }};
}

enum Opts {
    D4,
    D8,
    D16,
    S4,
    S8,
}

impl AdaptiveVec {
    fn choose_storage(len: usize, values: &[u32]) -> (Opts, usize) {
        let mut opt = Opts::D4;
        let mut min_size = Dense4::<u32>::estimate_size(len, values);

        let new_sz = DenseW::<u8, u32>::estimate_size(len, values);
        if new_sz < min_size {
            opt = Opts::D8;
            min_size = new_sz;
        }

        let new_sz = DenseW::<u16, u32>::estimate_size(len, values);
        if new_sz < min_size {
            opt = Opts::D16;
            min_size = new_sz;
        }

        let new_sz = CompressedIndexSparse::<Dense4<u32>>::estimate_size(len, values);
        if new_sz < min_size {
            opt = Opts::S4;
            min_size = new_sz;
        }

        let new_sz = CompressedIndexSparse::<DenseW<u8, u32>>::estimate_size(len, values);
        if new_sz < min_size {
            opt = Opts::S8;
        }

        (opt, min_size)
    }

    /// Create a new `AdaptiveVec` compressed sparse vector of length `len` for the given sparse data `values` and `indexes`.
    /// The most compact memory representation for the data is automatically selected.
    pub fn new(len: usize, values: &[u32], indexes: &[u32]) -> AdaptiveVec {
        let (opt, _) = AdaptiveVec::choose_storage(len, values);

        match opt {
            Opts::D4 => AdaptiveVec::D4(Dense4::construct(len, values, Some(indexes))),
            Opts::D8 => AdaptiveVec::D8(DenseW::<u8, u32>::construct(len, values, Some(indexes))),
            Opts::D16 => AdaptiveVec::D16(DenseW::<u16, u32>::construct(len, values, Some(indexes))),
            Opts::S4 => AdaptiveVec::S4(CompressedIndexSparse::<Dense4<u32>>::construct(
                len,
                values,
                Some(indexes),
            )),
            Opts::S8 => AdaptiveVec::S8(CompressedIndexSparse::<DenseW<u8, u32>>::construct(
                len,
                values,
                Some(indexes),
            )),
        }
    }

    /// Number of non-zero elements in the vector
    pub fn nnz(&self) -> usize {
        ada_expand!(self, v, v.nnz())
    }

    /// Total heap size occupied by vector
    pub fn mem_size(&self) -> usize {
        ada_expand!(self, v, v.mem_size())
    }

    /// Iterate over the non-zero elements of the vector
    pub fn iter(&self) -> AdaptiveVecIter {
        match self {
            AdaptiveVec::D4(v) => AdaptiveVecIter::D4(v.iter()),
            AdaptiveVec::D8(v) => AdaptiveVecIter::D8(v.iter()),
            AdaptiveVec::D16(v) => AdaptiveVecIter::D16(v.iter()),
            AdaptiveVec::S4(v) => AdaptiveVecIter::S4(v.iter()),
            AdaptiveVec::S8(v) => AdaptiveVecIter::S8(v.iter()),
        }
    }

    ///length
    pub fn len(&self) -> usize {
        match self {
            AdaptiveVec::D4(v) => v.len(),
            AdaptiveVec::D8(v) => v.len(),
            AdaptiveVec::D16(v) => v.len(),
            AdaptiveVec::S4(v) => v.len(),
            AdaptiveVec::S8(v) => v.len(),
        }
    }

    ///empty
    pub fn is_empty(&self) -> bool {
        match self {
            AdaptiveVec::D4(v) => v.is_empty(),
            AdaptiveVec::D8(v) => v.is_empty(),
            AdaptiveVec::D16(v) => v.is_empty(),
            AdaptiveVec::S4(v) => v.is_empty(),
            AdaptiveVec::S8(v) => v.is_empty(),
        }
    }

    /// Apply function `f` to the non-zero `(index, value)` tuples in the sparse vector.
    #[inline]
    pub fn foreach<F: FnMut(usize, u32)>(&self, mut f: F) {
        match self {
            AdaptiveVec::D4(v) => {
                for (idx, v) in v.iter() {
                    (f)(idx, v)
                }
            }
            AdaptiveVec::D8(v) => {
                for (idx, v) in v.iter() {
                    (f)(idx, v)
                }
            }
            AdaptiveVec::D16(v) => {
                for (idx, v) in v.iter() {
                    (f)(idx, v)
                }
            }
            AdaptiveVec::S4(v) => {
                for (idx, v) in v.iter() {
                    (f)(idx, v)
                }
            }
            AdaptiveVec::S8(v) => {
                for (idx, v) in v.iter() {
                    (f)(idx, v)
                }
            }
        }
    }

    /// dot
    pub fn dot(&self, rhs: &AdaptiveVec) -> u32 {
        let mut dot = 0;
        for merged in self
            .iter()
            .merge_join_by(rhs.iter(), |(lidx, _), (ridx, _)| lidx.cmp(ridx))
        {
            if let Both((_, lval), (_, rval)) = merged {
                dot += lval * rval;
            }
        }
        dot
    }

    /// to dense vector
    pub fn to_dense(&self) -> Array1<u32> {
        let len = self.len();
        let mut vec = vec![0u32; len];
        for (_, v) in self.iter().enumerate() {
            vec[v.0] = v.1;
        }
        arr1(&vec)
    }

    /// to dense vector, re-use storage
    pub(crate) fn to_vec(&self, vec: &mut Vec<u32>) {
        vec.clear();
        let len = self.len();
        vec.resize(len, 0);
        for v in self.iter() {
            vec[v.0] = v.1;
        }
    }
}

impl MemConstruct for AdaptiveVec {
    type Output = u32;

    fn estimate_size(len: usize, values: &[Self::Output]) -> usize {
        let (_, sz) = AdaptiveVec::choose_storage(len, values);
        sz
    }

    fn construct(len: usize, values: &[Self::Output], indexes: Option<&[u32]>) -> AdaptiveVec {
        AdaptiveVec::new(len, values, indexes.unwrap())
    }
}
/// Sparse Vector encoding `u32` values, with length up to `u32::MAX`.
/// Adaptively uses the most memory-efficient storage format based on
/// the data being stored.
#[derive(Debug)]
pub enum AdaptiveVecIter<'a> {
    /// Dense storage of 4-bit values, with fallback to a sparse array of u32 values
    D4(AbsIter<'a, Dense4<u32>>),

    /// Dense storage of 8-bit values, with fallback to a sparse array of u32 values
    D8(AbsIter<'a, DenseW<u8, u32>>),

    /// Dense storage of 16-bit values with fallback to a sparse array of u32 values
    D16(AbsIter<'a, DenseW<u16, u32>>),

    /// Compressed-index sparse storage, with values store as 4-bit dense array with u32 fallback.
    S4(AbsIter<'a, CompressedIndexSparse<Dense4<u32>>>),

    /// Compressed-index sparse storage, with values store as 4-bit dense array with u32 fallback.
    S8(AbsIter<'a, CompressedIndexSparse<DenseW<u8, u32>>>),
}

impl<'a> Iterator for AdaptiveVecIter<'a> {
    type Item = (usize, u32);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            AdaptiveVecIter::D4(v) => v.next(),
            AdaptiveVecIter::D8(v) => v.next(),
            AdaptiveVecIter::D16(v) => v.next(),
            AdaptiveVecIter::S4(v) => v.next(),
            AdaptiveVecIter::S8(v) => v.next(),
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::gen_rand::gen_vec_bounded;
    use rand::prelude::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;
    use std::fmt::Debug;
    use std::ops::Range;

    fn test_sparse<T, R: Rng>(rng: &mut R, len: usize, values: &[u32], indexes: &[u32])
    where
        T: AbstractVec<Output = u32> + Debug + MemConstruct<Output = u32>,
        T::Index: Debug,
    {
        let arr = T::construct(len, values, Some(indexes));

        // Check that every sparse value is present for
        for (idx, v) in indexes.iter().zip(values) {
            if arr.get(*idx as usize) != *v {
                println!("values: {values:?}");
                println!("indexes: {indexes:?}");
                println!("dd: {arr:?}");
            }
            assert_eq!(arr.get(*idx as usize), *v)
        }

        // Check that every index gives the expected result
        for i in 0..len {
            let r = arr.get(i);

            match indexes.binary_search(&(i as u32)) {
                Ok(v) => assert_eq!(r, values[v]),
                Err(_) => assert_eq!(r, 0),
            }
        }

        // Iterate over non-zero elements and make sure they match
        for (count, (idx, v)) in arr.iter().enumerate() {
            if indexes[count] != idx as u32 || values[count] != v {
                println!("count: {count:?}");
                println!("values: {values:?}");
                println!("indexes: {indexes:?}");
                println!("dd: {arr:?}");
            }

            assert_eq!(indexes[count], idx as u32);
            assert_eq!(values[count], v);
        }

        // Iterate over some ranges & make sure they're valid
        for _ in 0..10 {
            let range = gen_range(rng, len);
            let sparse_range = convert_range(range.clone(), indexes);

            let values = &values[sparse_range.clone()];
            let indexes = &indexes[sparse_range.clone()];

            let mut got_all = false;
            let mut n = 0;

            for (count, (idx, v)) in arr.iter_range(range.clone()).enumerate() {
                if count >= indexes.len() || indexes[count] != idx as u32 || values[count] != v {
                    println!("count: {count:?}");
                    println!("idx: {idx}");
                    println!("v: {v}");
                    println!("values: {values:?}");
                    println!("indexes: {indexes:?}");
                    println!("dd: {arr:?}");
                }

                assert_eq!(indexes[count], idx as u32);
                assert_eq!(values[count], v);

                n += 1;
            }

            if n == indexes.len() as i32 {
                got_all = true;
            }
            assert!(got_all);
        }
    }

    pub fn convert_range(pos_range: Range<usize>, indexes: &[u32]) -> Range<usize> {
        let start = indexes.binary_search(&(pos_range.start as u32));
        let end = indexes.binary_search(&(pos_range.end as u32));

        let s = match start {
            Ok(v) => v,
            Err(v) => v,
        };
        let e = match end {
            Ok(v) => v,
            Err(v) => v,
        };
        s..e
    }

    pub fn gen_range(rng: &mut impl Rng, len: usize) -> Range<usize> {
        let start: usize = rng.gen_range(0..len);
        let end: usize = rng.gen_range(start..len);
        start..end
    }

    fn test_sparse_many<'a, T: 'a + AbstractVec<Output = u32> + Debug + MemConstruct<Output = u32>>()
    where
        T::Index: Debug,
    {
        let mut rng = Pcg64Mcg::seed_from_u64(42);

        for i in (0..500).step_by(8) {
            let vec_len: usize = rng.gen_range(0..100000);
            let nnz: usize = if vec_len == 0 { 0 } else { rng.gen_range(0..vec_len) };
            let mut indexes: Vec<u32> = gen_vec_bounded(&mut rng, nnz, vec_len as u32);
            indexes.sort_unstable();
            indexes.dedup();

            let values = gen_vec_bounded(&mut rng, indexes.len(), 10 + 10 * i);

            test_sparse::<T, _>(&mut rng, vec_len, &values, &indexes);
        }
    }

    #[test]
    fn run_test() {
        test_sparse_many::<Dense4<u32>>();
        test_sparse_many::<DenseW<u8, u32>>();
        test_sparse_many::<DenseW<u16, u32>>();

        test_sparse_many::<SimpleSparse<u32>>();
        test_sparse_many::<CompressedIndexSparse<Dense4<u32>>>();
        test_sparse_many::<CompressedIndexSparse<SimpleSparse<u32>>>();
    }
    #[test]
    fn test_dot() {
        let vec1 = AdaptiveVec::new(10, &[2, 3, 4], &[1, 2, 3]);
        let vec2 = AdaptiveVec::new(10, &[2, 4], &[1, 3]);
        let res = vec1.dot(&vec2);
        print!(" dot {res:?}");
        assert_eq!(res, vec1.to_dense().dot(&vec2.to_dense()))
    }
}
