pub mod hs_tree;
mod gats;
pub use hs_tree::HSTree;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
/// Structure that allows searching for the `k` closest strings to a query by Levenshtein distance
pub struct StringSearcher {
    pub tree: HSTree,
}

/// Structure that associates a string with its Levenshtein distance from the query
#[derive(PartialEq, Eq, PartialOrd)]
pub struct Result {
    pub data: String,
    pub distance: u32,
}

impl Ord for Result {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}

/// Structure that holds the closest and at most `limit` results
struct ResultRankings {
    best: BinaryHeap<Result>,
    limit: u32,
}

impl ResultRankings {
    /// Returns a threshold for the Levenshtein distance for a string to be placed into the best results
    fn threshold(&self) -> u32 {
        // we don't have enough results yet to limit our Levenshtein distance
        if self.best.len() < self.limit as usize {
            u32::MAX
        }
        // we won't accept any distance worse than the worst so far in the top-k
        else {
            // this will fail if best.len() is 0 and limit is also 0
            let worst = best.peek().unwrap();
            worst.distance
        }
    }
    fn new(limit: u32) -> ResultRankings {
        ResultRankings { best: Default::default(), limit }
    }
}

impl StringSearcher {
    /// Returns up to `limit` strings with the closest Levenshtein distance to `query`,
    /// The result is sorted by ascending Levenshtein distance
    pub fn top(&self, query: &str, limit: u32) -> Vec<Result> {
        let mut best = ResultRankings::new(limit);
        let mut upper_bound = u32::MAX;
        for i in 0..query.len() {
            if (1 << i) >= (upper_bound + 1) {
                break;
            }
            // saturating sub/add prevents under/overflow
            for length in (query.len().saturating_sub(upper_bound))..=(query.len().saturating_add(upper_bound)) {
                if let Some(group) = &self.tree.groups.get(length) {

                }
            }
        }
        best.into_sorted_vec()
    }
}