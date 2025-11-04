use crate::dist::{DistanceFn, DistanceType, Q};
use log::warn;
use ndarray::{s, Array2, Axis};
use noisy_float::checkers::NumChecker;
use noisy_float::NoisyFloat;
use num_traits::Bounded;
use rayon::prelude::*;
use std::fmt::Debug;
use vpsearch::{BestCandidate, MetricSpace, Tree};

#[derive(Clone, Debug)]
pub struct Sample<'a> {
    data: &'a [Q],
    pub idx: usize,
}

impl<'a> Sample<'a> {
    fn new(data: &'a [Q], idx: usize) -> Self {
        Sample { data, idx }
    }
}

/// `MetricSpace` makes items comparable. It's a bit like Rust's `PartialOrd`.
impl vpsearch::MetricSpace for Sample<'_> {
    type UserData = DistanceFn;
    type Distance = NoisyFloat<Q, NumChecker>;

    fn distance(&self, other: &Self, dist: &Self::UserData) -> Self::Distance {
        NoisyFloat::new(dist(self.data, other.data) as Q)
    }
}

/// Add custom search for finding the index of the N nearest points
struct CountBasedNeighborhood<Item, Impl>
where
    Item: MetricSpace<Impl>,
    Item::Distance: Ord,
{
    // Max amount of items
    max_item_count: usize,
    // The max distance we have observed so far
    max_observed_distance: Item::Distance,
    // A list of indexes no longer than max_item_count sorted by distance
    distance_x_index: Vec<(Item::Distance, usize)>,
}

impl<Item, Impl> CountBasedNeighborhood<Item, Impl>
where
    Item: MetricSpace<Impl>,
    Item::Distance: Ord,
{
    /// Helper function for creating the CountBasedNeighborhood struct.
    /// Here `item_count` is the amount of items returned, the k in knn.
    fn new(max_item_count: usize) -> Self {
        CountBasedNeighborhood {
            max_item_count,
            max_observed_distance: <Item::Distance as Bounded>::max_value(),
            distance_x_index: Vec::<(Item::Distance, usize)>::with_capacity(max_item_count + 1),
        }
    }

    /// Clear the contents neighborhood
    fn clear(&mut self) {
        self.max_observed_distance = <Item::Distance as Bounded>::max_value();
        self.distance_x_index.clear();
    }

    /// Insert a single index in the correct possition given that the
    /// `distance_x_index` is already sorted.
    fn insert_index(&mut self, index: usize, distance: Item::Distance) {
        let val = (distance, index);
        let idx = self.distance_x_index.binary_search(&val).unwrap_or_else(|x| x);
        self.distance_x_index.insert(idx, val);
        if self.distance_x_index.len() >= self.max_item_count {
            self.distance_x_index.truncate(self.max_item_count);
            self.max_observed_distance = self.distance_x_index.last().unwrap().0;
        }
    }
}

/// Best candidate definitions that tracks of the index all the points
/// within the radius of `distance` as specified in the `RadiusBasedNeighborhood`.
impl<'a, Item, Impl> BestCandidate<Item, Impl> for &'a mut CountBasedNeighborhood<Item, Impl>
where
    Item: MetricSpace<Impl> + Clone,
    Item::Distance: Ord,
{
    type Output = std::iter::Cloned<std::slice::Iter<'a, (Item::Distance, usize)>>;

    #[inline]
    fn consider(&mut self, _: &Item, distance: Item::Distance, candidate_index: usize, _: &Item::UserData) {
        // Early out, no need to do track any points if the max return size is 0
        if self.max_item_count == 0 {
            return;
        }

        if distance < self.max_observed_distance || self.distance_x_index.len() < self.max_item_count {
            self.insert_index(candidate_index, distance);
        }
    }

    #[inline]
    fn distance(&self) -> Item::Distance {
        self.max_observed_distance
    }

    fn result(self, _: &Item::UserData) -> Self::Output {
        self.distance_x_index.as_slice().iter().cloned()
    }
}

pub fn nearest_neighbors(
    data: &Array2<Q>,
    mut k: usize,
    distance_type: DistanceType,
    num_threads: usize,
) -> (Array2<usize>, Array2<Q>) {
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let (samples_size, _) = data.dim();
    if samples_size <= k {
        warn!("{} neighbors requested, but only {} available", k, samples_size - 1);
        k = samples_size - 1;
    }

    let metric = distance_type.metric();
    let m2d = distance_type.metric2dist();

    let samples = data
        .axis_iter(Axis(0))
        .enumerate()
        .map(|(idx, _)| Sample::new(data.slice(s![idx, ..]).to_slice().unwrap(), idx))
        .collect::<Vec<Sample>>();
    let vp = Tree::new_with_user_data_ref(&samples, &metric);

    let mut indices = Array2::from_elem((samples_size, k), usize::MAX);
    let mut distances = Array2::from_elem((samples_size, k), Q::INFINITY);

    thread_pool.install(|| {
        indices
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip_eq(distances.axis_iter_mut(Axis(0)).into_par_iter())
            .enumerate()
            .for_each_init(
                || CountBasedNeighborhood::new(k + 1),
                |neighborhood, (cell, (mut indices, mut distances))| {
                    neighborhood.clear();
                    let query = &samples[cell];
                    let mut j = 0;
                    for (dist, idx) in vp.find_nearest_custom(query, &metric, neighborhood) {
                        if query.idx != idx && j < k {
                            indices[j] = idx;
                            distances[j] = m2d(dist.raw());
                            j += 1;
                        }
                    }
                },
            );
    });

    (indices, distances)
}
