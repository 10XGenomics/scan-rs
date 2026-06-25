use anyhow::{format_err, Error};
use serde::{self, Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct LabelClass {
    pub labels: Vec<String>,
    pub offsets: Vec<i64>,
    #[serde(default, with = "indices_serde")]
    pub indices: Vec<i64>,
}

mod indices_serde {
    use super::{Deserialize, Serialize};
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(value: &Vec<i64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Some(value).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<i64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<Vec<i64>>::deserialize(deserializer).map(Option::unwrap_or_default)
    }
}

impl LabelClass {
    pub fn new(labels: Vec<String>, offsets: Vec<i64>, indices: Vec<i64>) -> Result<LabelClass, Error> {
        if labels.len() != offsets.len() {
            return Err(format_err!("Label and offsets length unequal"));
        }
        Ok(LabelClass {
            labels,
            offsets,
            indices,
        })
    }

    pub fn blank() -> LabelClass {
        LabelClass {
            labels: Vec::new(),
            offsets: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Attach indices to the label class if not already done.
    pub fn attach_indices(&mut self, indices: Vec<i64>) -> Result<bool, Error> {
        if !self.indices.is_empty() {
            return Err(format_err!("Indices already attached to LabelClass object."));
        }
        self.indices = indices;
        Ok(true)
    }

    /// Get the index of a label string. Returns None if the label string is not found.
    fn get_label_index(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|x| x == label)
    }

    /// Returns either the set of indices pointing to the given label or None is the label doesn't exist.
    pub fn get_indices(&self, label: &str) -> Option<&[i64]> {
        self.get_label_index(label).map(|idx| {
            let offset = self.offsets[idx] as usize;
            if idx == self.offsets.len() - 1 {
                &self.indices[offset..]
            } else {
                let next_offset = self.offsets[idx + 1] as usize;
                &self.indices[offset..next_offset]
            }
        })
    }

    /// Remove a feature types from the index corresponding to `pattern`. Returns indices of said features.
    pub fn remove_like(&mut self, pattern: &str) -> BTreeSet<usize> {
        let mut r = BTreeSet::default();
        let mut idx = 0;
        while idx < self.labels.len() {
            if self.labels[idx].contains(pattern) {
                self.remove_index(idx, &mut r);
            } else {
                idx += 1;
            }
        }
        r
    }

    /// Remove features types unlike the pattern. Also, return their indices.
    pub fn remove_unlike(&mut self, pattern: &str) -> BTreeSet<usize> {
        let mut r = BTreeSet::default();
        let mut idx = 0;
        while idx < self.labels.len() {
            if self.labels[idx].contains(pattern) {
                idx += 1;
            } else {
                self.remove_index(idx, &mut r);
            }
        }
        r
    }

    // TODO: I tried -> Splice<'a, impl Iterator<Item = i64> + 'a>, but it crashed rustc, which was exciting
    fn remove_index(&mut self, idx: usize, set: &mut BTreeSet<usize>) {
        self.labels.remove(idx);
        let start = self.offsets.remove(idx);
        let end = if self.offsets.len() > idx {
            let end = self.offsets[idx];
            let len = end - start;
            for v in self.offsets.iter_mut().skip(idx) {
                *v -= len;
            }
            end as usize
        } else {
            self.indices.len()
        };
        set.extend(self.indices.splice(start as usize..end, vec![]).map(|v| v as usize));
    }
}

/*
Code used for working with CountMatrices, moved here be shared with PyO3 and hdf5_io
 */
pub fn make_labelclass_from_feature_type_vector(feature_types: &[String]) -> Result<LabelClass, Error> {
    let mut idx = 0;
    let mut labels = vec![feature_types
        .first()
        .ok_or_else(|| format_err!("no features found!"))?
        .clone()];
    let mut offsets = vec![idx as i64];
    for (i, feature_type) in feature_types.iter().enumerate() {
        if feature_type != &feature_types[idx] {
            idx = i;
            labels.push(feature_types[idx].clone());
            offsets.push(idx as i64);
        }
    }
    let indices = (0..feature_types.len()).map(|v| v as i64).collect::<Vec<_>>();
    LabelClass::new(labels, offsets, indices)
}

#[cfg(test)]
mod test {
    use super::*;
    use bincode::{deserialize, serialize};

    fn sample_label_class(indices: bool) -> LabelClass {
        LabelClass {
            labels: vec!["a", "b"].into_iter().map(ToString::to_string).collect(),
            offsets: vec![0, 2],
            indices: if indices { vec![3, 5, 8] } else { vec![] },
        }
    }

    #[test]
    fn test_remove_index() {
        let x = LabelClass {
            labels: vec!["a", "b", "c", "d"]
                .into_iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            offsets: vec![0, 10, 21, 34],
            indices: (0..50).collect::<Vec<_>>(),
        };

        let mut xa = x.clone();
        let ba = xa.remove_like("a");
        assert_eq!(ba, (0..10).collect::<BTreeSet<_>>());
        assert_eq!(xa.labels, vec!["b", "c", "d"]);
        assert_eq!(xa.offsets, vec![0, 11, 24]);
        assert_eq!(xa.indices, (10..50).collect::<Vec<_>>());

        let mut xb = x.clone();
        let bb = xb.remove_like("b");
        assert_eq!(bb, (10..21).collect::<BTreeSet<_>>());
        assert_eq!(xb.labels, vec!["a", "c", "d"]);
        assert_eq!(xb.offsets, vec![0, 10, 23]);
        let mut indices = (0..10).collect::<Vec<_>>();
        indices.extend(21..50);
        assert_eq!(xb.indices, indices);

        let mut xc = x.clone();
        let bc = xc.remove_like("c");
        assert_eq!(bc, (21..34).collect::<BTreeSet<_>>());
        assert_eq!(xc.labels, vec!["a", "b", "d"]);
        assert_eq!(xc.offsets, vec![0, 10, 21]);
        let mut indices = (0..21).collect::<Vec<_>>();
        indices.extend(34..50);
        assert_eq!(xc.indices, indices);

        let mut xd = x;
        let bd = xd.remove_like("d");
        assert_eq!(bd, (34..50).collect::<BTreeSet<_>>());
        assert_eq!(xd.labels, vec!["a", "b", "c"]);
        assert_eq!(xd.offsets, vec![0, 10, 21]);
        assert_eq!(xd.indices, (0..34).collect::<Vec<_>>());
    }

    #[test]
    fn test_indices_deserialize_null_as_empty() {
        // null indices should deserialize as empty vec
        let json = r#"{"labels": [], "offsets": [], "indices": null}"#;
        let lc: LabelClass = serde_json::from_str(json).expect("failed to deserialize null indices");
        assert_eq!(lc.indices, Vec::<i64>::new());
    }

    #[test]
    fn test_indices_deserialize_missing_as_default() {
        // missing indices field should deserialize as empty vec
        let json = r#"{"labels": [], "offsets": []}"#;
        let lc: LabelClass = serde_json::from_str(json).expect("failed to deserialize missing indices");
        assert_eq!(lc.indices, Vec::<i64>::new());
    }

    #[test]
    fn test_json_roundtrip_indices() {
        let label_class = sample_label_class(true);
        let json = serde_json::to_string(&label_class).unwrap();
        assert_eq!(json, r#"{"labels":["a","b"],"offsets":[0,2],"indices":[3,5,8]}"#);
        let round_trip: LabelClass = serde_json::from_str(&json).unwrap();

        assert_eq!(round_trip.labels, label_class.labels);
        assert_eq!(round_trip.offsets, label_class.offsets);
        assert_eq!(round_trip.indices, label_class.indices);
    }

    #[test]
    fn test_json_roundtrip_no_indices() {
        let label_class = sample_label_class(false);
        let json = serde_json::to_string(&label_class).unwrap();
        assert_eq!(json, r#"{"labels":["a","b"],"offsets":[0,2],"indices":[]}"#);
        let round_trip: LabelClass = serde_json::from_str(&json).unwrap();

        assert_eq!(round_trip.labels, label_class.labels);
        assert_eq!(round_trip.offsets, label_class.offsets);
        assert_eq!(round_trip.indices, label_class.indices);
    }

    #[test]
    fn test_bincode_roundtrip_indices() {
        let label_class = sample_label_class(true);
        let encoded = serialize(&label_class).unwrap();
        let round_trip: LabelClass = deserialize(&encoded).unwrap();

        assert_eq!(round_trip.labels, label_class.labels);
        assert_eq!(round_trip.offsets, label_class.offsets);
        assert_eq!(round_trip.indices, label_class.indices);
    }

    #[test]
    fn test_bincode_roundtrip_no_indices() {
        let label_class = sample_label_class(false);
        let encoded = serialize(&label_class).unwrap();
        let round_trip: LabelClass = deserialize(&encoded).unwrap();

        assert_eq!(round_trip.labels, label_class.labels);
        assert_eq!(round_trip.offsets, label_class.offsets);
        assert_eq!(round_trip.indices, label_class.indices);
    }
}
