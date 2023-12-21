mod gats;
mod prefix;
use std::{cmp::Ordering, fmt::Display};

/*
mod hs_tree;
use hs_tree::HSTree;
use hs_tree::Rankings;
*/

/// Structure that associates a string with its Levenshtein distance from the query
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct MeasuredString {
    pub string: String,
    pub distance: usize,
}
impl Ord for MeasuredString {
    /// Compare the edit distances for MeasuredString
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}

/// Structure that associates a string with its its prefix edit distance from the query
#[derive(PartialEq, Eq, Debug)]
pub struct MeasuredPrefix {
    pub string: String,
    pub prefix_distance: usize,
}
impl Ord for MeasuredPrefix {
    /// Compare the prefix and then full edit distances for MeasuredPrefix
    fn cmp(&self, other: &Self) -> Ordering {
        self.prefix_distance
            .cmp(&other.prefix_distance)
    }
}
impl PartialOrd for MeasuredString {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialOrd for MeasuredPrefix {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub type Autocompleter<'stored, U> = prefix::Trie<'stored, U>;
//pub type StringSearcher<'a, U> = HSTree<'a, U>;
