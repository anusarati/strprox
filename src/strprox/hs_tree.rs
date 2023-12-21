/// Implementation of the HS-Tree and HS-Topk algorithm from doi:10.1109/ICDE.2015.7113311
///
/// Uses the CPMerge algorithm from "Simple and Efficient Algorithm for Approximate Dictionary Matching" by Okazaki and Tsujii
///
/// Uses the Length-aware edit distance verification algorithm from doi:10.14778/2078331.2078340 referenced in the HSTree paper
use bitvec::prelude::*;
use ndarray::{Array1, Array2};
use std::cmp::Ordering;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::{btree_map, BinaryHeap};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Add, AddAssign, Range};
//use crate::strprox::traits::StringSizeType;

type Superstrings<'stored> = HashSet<&'stored str>;
type InvertedIndex<'stored> = HashMap<&'stored str, Superstrings<'stored>>;

/// Type that bounds the length of strings stored in the tree
//type U = u16;

/// Contains a map from substrings to original strings
///
/// The substrings are restricted to a certain length and position based on the location in the tree
#[derive(Clone, Default)]
pub struct HSTreeNode<'stored> {
    index: InvertedIndex<'stored>,
}

impl<'stored> HSTreeNode<'stored> {
    // Inserts an association between a substring and a string containing it
    fn insert(&mut self, substring: &'stored str, string: &'stored str) {
        if let Some(strings) = self.index.get_mut(substring) {
            strings.insert(string);
        } else {
            let mut strings = Superstrings::<'stored>::new();
            strings.insert(string);
            self.index.insert(substring, strings);
        }
    }
}
#[derive(PartialEq, Eq, Hash, Clone)]
struct SegmentRange<U> {
    range: Range<U>,
}
type Frequencies<'stored, U> = HashMap<&'stored str, U>;
#[derive(Clone, Default)]
pub struct HSLevel<'stored, U> {
    // this should be at most the length of the original string, so iteration shouldn't be prohibitive
    nodes: Vec<HSTreeNode<'stored>>,
    /// The starting indices of the superstring slices corresponding to the substrings in each node
    start_positions: Vec<U>,
    /// Stores frequencies of substrings in this level (for CPMerge)
    frequencies: Frequencies<'stored, U>,
}
#[derive(Clone, Copy)]
struct LevelCoordinates<U> {
    depth: U,
    /// Length of original string
    length: U,
}

/// enum for whether a node has the smaller or larger length in a level
#[derive(PartialEq, Eq, Clone, Copy)]
enum LevelLengthCategory {
    Lesser = 0,
    Greater = 1,
}
use LevelLengthCategory::{Greater, Lesser};

use super::MeasuredString;
/// Represents the segmentation of a query string to compare with the substrings at a certain tree level
struct LevelSegments<U> {
    /// Starting indices of each segment sorted in order of ascending match frequency (CPMerge)
    ///
    /// A level in the tree has substrings of two possible lengths
    /// The first vector contains the indices for the shorter segments
    /// The second vector contains the indices for the longer segments
    segments: [Vec<U>; 2],
}
//type SegmentRangeSet = BTreeSet<SegmentRange<U>>;
/// Maps a query segment's range to a matching tree string segment range
/// sorting only required for finding maximum disjoint set of ranges
type SegmentMatches<U> = BTreeMap<SegmentRange<U>, SegmentRange<U>>;
//type UnorderedSegmentRangeSet = HashSet<SegmentRange<U>>;

type MatchSegmentMap<'stored, U> = HashMap<&'stored str, SegmentMatches<U>>;
//type UnorderedMatchSegmentMap<'stored> = HashMap<&'stored str, UnorderedSegmentMatches>;
type FrequencyMap<'stored, U> = BTreeMap<U, StringSegmentMatches<'stored, U>>;
#[derive(Default)]
/// Maps parent strings in the tree to a query's matching segment ranges
struct StringSegmentMatches<'stored, U> {
    matches: MatchSegmentMap<'stored, U>,
}
struct MatchFinder<'stored, 'search, U> {
    coords: LevelCoordinates<U>,
    level: &'stored HSLevel<'stored, U>,
    query: &'search str,
    threshold: U,
    segments: LevelSegments<U>,
}
type LengthGroups<'stored, U> = BTreeMap<U, HSLengthGroup<'stored, U>>;
type Levels<'stored, U> = Vec<HSLevel<'stored, U>>;

/// Represents a group of strings with the same length in the tree
pub struct HSLengthGroup<'stored, U> {
    // avoid explicit branch representation to allow for efficient traversal
    levels: Levels<'stored, U>,
    /// The full strings inside the group
    original: Superstrings<'stored>,
    length: U,
}

/// Structure that holds at most `limit` of the strings closest to the query
#[derive(Clone, Default)]
pub struct Rankings<U: Ord> {
    best: BinaryHeap<MeasuredString<U>>,
    limit: U,
    initial_threshold: U,
}
/// Provides interface to select the next closest length to search within the tree
struct LengthSelector<U> {
    lower_range: Range<U>,
    upper_range: Range<U>,
    query_len: U,
}
#[derive(Default)]
/// Structure that allows for fast queries for the top closest strings using filters on length and position
pub struct HSTree<'stored, U> {
    // indexed by length, then depth, then sibling number
    groups: LengthGroups<'stored, U>,
}

type UUU = u8;
/*
macro_rules! impl_hs_tree {
    ($(UUU:ty)*) => {
        $(
            */
impl PartialOrd for SegmentRange<UUU> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Order SegmentRanges based on the order of their starting points, then their ending points
impl Ord for SegmentRange<UUU> {
    fn cmp(&self, other: &Self) -> Ordering {
        // not using range as Iterator's cmp because it would iterate through the range
        let start_order = self.range.start.cmp(&other.range.start);
        if start_order != Ordering::Equal {
            return start_order;
        }
        self.range.end.cmp(&other.range.end)
    }
}

impl From<Range<UUU>> for SegmentRange<UUU> {
    fn from(value: Range<UUU>) -> Self {
        SegmentRange { range: value }
    }
}

impl From<Range<usize>> for SegmentRange<UUU> {
    fn from(value: Range<usize>) -> Self {
        let Range::<usize> { start, end } = value;
        let range: Range<UUU> = start as UUU..end as UUU;
        range.into()
    }
}

impl From<SegmentRange<UUU>> for Range<UUU> {
    fn from(value: SegmentRange<UUU>) -> Self {
        value.range
    }
}

impl From<SegmentRange<UUU>> for Range<usize> {
    fn from(value: SegmentRange<UUU>) -> Self {
        let Range::<UUU> { start, end } = value.range;
        start as usize..end as usize
    }
}

impl SegmentRange<UUU> {
    /// Splits the SegmentRange in half, with the left having at most the size of the right
    fn split(self) -> (Self, Self) {
        let range: Range<UUU> = self.into();
        let mid = (range.start + range.end) / 2;
        let left = range.start..mid;
        let right = mid..range.end;
        (left.into(), right.into())
    }
}

impl HSLevel<'_, UUU> {
    fn new(length: UUU) -> Self {
        let length = length as usize;
        let mut nodes = Vec::<HSTreeNode>::with_capacity(length);
        nodes.resize(length, Default::default());
        let mut start_positions = Vec::<UUU>::with_capacity(length + 1);
        start_positions.resize(length + 1, 0);
        Self {
            nodes,
            start_positions,
            frequencies: Default::default(),
        }
    }
    /// Returns the starting position of the substrings at `index` inside their superstrings
    fn substring_start_pos(&self, index: UUU) -> UUU {
        self.start_positions[index as usize]
    }
    /// Returns the node index of at least `lower` of the substring beginning at `start`
    fn index(&self, start: UUU, lower: UUU) -> UUU {
        self.start_positions[lower as usize..]
            .binary_search(&start)
            .expect("`start_positions` should be sorted, and `start` should exist inside it")
            as UUU
    }
    /// Returns the SegmentRange associated with the substrings at the node at `index`
    fn substring_range(&self, index: UUU) -> SegmentRange<UUU> {
        let start = self.substring_start_pos(index);
        let end = self.substring_start_pos(index + 1);
        let range = start..end;
        range.into()
    }
    /// Returns the length of substrings at the node at `index`
    fn substring_len(&self, index: UUU) -> UUU {
        let Range::<UUU> { start, end } = self.substring_range(index).into();
        end - start
    }
    /// Computes range of indices to query segments that may match a substring in the node at `index`
    fn candidate_range(&self, query: &str, threshold: UUU, index: UUU) -> SegmentRange<UUU> {
        let pos = self.substring_start_pos(index);
        let len = self.substring_len(index);
        let delta = query.len() as UUU - len;
        // `pos` and node `index` begin at 0, unlike in the paper (evidence: lower was at least 1 instead of 0 in the paper)
        // the edit distance `threshold` and length difference `delta` are starting-index invariant
        let lower = max(0, max(pos - index, pos + delta - threshold + index));
        let upper = min(delta, min(pos + index, pos + delta + threshold - index));
        let range = lower..(upper + 1);
        range.into()
    }
}

impl LevelCoordinates<UUU> {
    /// Returns the minimum possible length of a substring in this level
    fn lesser_len(&self) -> UUU {
        self.depth / (1 << self.length)
    }
    /// Returns the maximum possible length of a substring in this level
    fn greater_len(&self) -> UUU {
        self.lesser_len() + 1
    }
    /// Returns the minimum and maximum lengths of substrings in this level
    fn lengths(&self) -> (UUU, UUU) {
        let lesser = self.lesser_len();
        (lesser, lesser + 1)
    }
    /// Categorizes the length as Lesser or Greater (Greater by default)
    fn categorize_len(&self, len: UUU) -> LevelLengthCategory {
        if len == self.lesser_len() {
            Lesser
        } else {
            Greater
        }
    }
}

impl LevelSegments<UUU> {
    /// Returns vector of frequencies for each query segment
    /// `segments`: number of segments
    /// `len`: length of each segment
    fn frequencies(
        segments: UUU,
        len: UUU,
        query: &str,
        frequencies: &Frequencies<UUU>,
    ) -> Vec<UUU> {
        (0..segments)
            .map(|start| {
                let start = start as usize;
                let len = len as usize;
                *frequencies.get(&query[start..start + len]).unwrap_or(&0)
            })
            .collect()
    }

    /// Produces segmentation information for `query` based on the frequencies of each query segment
    /// at a certain tree level
    fn new(
        query: &str,
        depth: UUU,
        length: UUU,
        frequencies: &Frequencies<UUU>,
    ) -> LevelSegments<UUU> {
        // minimum length of a substring at the level given by depth and length in the tree
        let min_len = length / (1 << depth);
        let mut segments: [Vec<UUU>; 2] = Default::default();
        // number of shorter substring segments
        let n_shorter = query.len() as UUU - min_len + 1;
        // length of substrings in a level can differ by at most 1
        let n_longer = n_shorter - 1;
        segments[0] = (0..n_shorter).collect();
        // longer substring segments
        segments[1] = (0..query.len() as UUU - min_len).collect();

        let mut segment_frequencies: [Vec<UUU>; 2] = Default::default();
        segment_frequencies[0] =
            LevelSegments::<UUU>::frequencies(n_shorter, min_len, query, frequencies);
        segment_frequencies[1] =
            LevelSegments::<UUU>::frequencies(n_longer, min_len + 1, query, frequencies);
        for i in 0..2 {
            let segment_frequencies = &segment_frequencies[i];
            // sort the segment starting positions from those with least associated substrings to those with the most
            // for CPMerge
            segments[i].sort_by(|a, b| {
                segment_frequencies[*a as usize].cmp(&segment_frequencies[*b as usize])
            });
        }

        LevelSegments { segments }
    }
    fn at(&self, len_category: LevelLengthCategory, index: UUU) -> UUU {
        self.segments[len_category as usize][index as usize]
    }
}

impl<'stored> StringSegmentMatches<'stored, UUU> {
    /// Returns the maximum disjoint set from the `ranges` for matching query segments
    /// Used to replace SEGCOUNT from doi:10.1109/ICDE.2015.7113311
    fn maximum_disjoint_set(ranges: SegmentMatches<UUU>) -> SegmentMatches<UUU> {
        // Algorithm cited from doi:10.1016/s0925-7721(98)00028-5
        // https://en.wikipedia.org/wiki/Maximum_disjoint_set#1-dimensional_intervals:_exact_polynomial_algorithm
        // This results in the maximum disjoint set, so dynamic programming is unnecessary to find the greatest number of non-overlapping matches
        // However, for a small number of ranges, using the DP SEGCOUUUNT may be faster than using the sorted BTree

        // track the current upper bound used to prune overlapping ranges
        // using i32 limits the lengths of strings to i32::MAX
        let mut upper: i64 = -1;
        let mut ranges = ranges;
        ranges.retain(|query_range, _| {
            let query_range: Range<UUU> = (*query_range).into();
            // skip all overlapping ranges (the first range with the end at upper cannot intersect more than the others)
            if query_range.start as i64 > upper {
                upper = query_range.end as i64;
                return true;
            }
            false
        });
        ranges
    }

    /// Inserts a matching pair of substring ranges from the query and stored `string`
    /// `subquery_range`: range associated with query string segment
    /// `substring_range`: range associated with stored string segment
    fn insert(
        &mut self,
        string: &'stored str,
        subquery_range: SegmentRange<UUU>,
        substring_range: SegmentRange<UUU>,
    ) {
        if let Some(ranges) = self.matches.get_mut(&string) {
            ranges.insert(subquery_range, substring_range);
        } else {
            let mut ranges = SegmentMatches::default();
            ranges.insert(subquery_range, substring_range);
            self.matches.insert(string, ranges);
        }
    }
    /// Associate the `subquery_range` and `substring_range` with the matches of all substrings already matched that are also in the node
    /// if the node has matches for the query substring
    /// `substring_range`: the range associated with all substrings in the node's inverted index
    fn merge(
        &mut self,
        node: &HSTreeNode,
        subquery_range: SegmentRange<UUU>,
        query: &str,
        substring_range: SegmentRange<UUU>,
    ) {
        let slice_range: Range<usize> = subquery_range.clone().into();
        let query_substring: &str = &query[slice_range];
        if let Some(superstrings) = node.index.get(query_substring) {
            for (string, ranges) in &mut self.matches {
                if superstrings.contains(string) {
                    ranges.insert(subquery_range.clone(), substring_range.clone());
                }
            }
        }
    }
    /// Removes all matches for the string
    fn remove(&mut self, string: &str) {
        self.matches.remove(string);
    }
    /// Remove all strings that do not have at least `minimum_matches`
    fn filter(&mut self, minimum_matches: UUU) {
        self.matches.retain(|string, ranges| {
            *ranges = Self::maximum_disjoint_set(*ranges);
            (ranges.len() as UUU) >= minimum_matches
        });
    }
    /// Remove all overlapping query segments in the matches associated with each stored string
    /// Strings that no longer have any matches are removed from the map
    fn remove_overlaps(&mut self) {
        self.filter(1);
    }
    /// Consumes the matches and returns a map from match frequency to match information for the superstring
    fn to_sorted(mut self) -> FrequencyMap<'stored, UUU> {
        let mut frequency_map = FrequencyMap::new();
        self.remove_overlaps();
        for (string, ranges) in self.matches {
            let frequency = ranges.len() as UUU;
            let mapped: &mut StringSegmentMatches<UUU>;
            if let Some(strings) = frequency_map.get_mut(&frequency) {
                mapped = strings;
            } else {
                frequency_map.insert(frequency, Default::default());
                mapped = frequency_map.get_mut(&frequency).unwrap();
            }
            mapped.matches.insert(string, ranges);
        }
        frequency_map
    }
}

impl<'stored, 'search> MatchFinder<'stored, 'search, UUU> {
    /// Adds substring matches within a level
    /// `cond` is used to limit the range of the iteration differently between the parts of the level with different lengths of substrings
    /// `insert` is expected to modify a captured SegmentMatches struct based on matches from a node's inverted index information
    fn crawl<'a, CondFn, InsertFn>(
        &'a mut self,
        segment_index: UUU,
        mut cond: CondFn,
        mut insert: InsertFn,
    ) where
        CondFn: FnMut(LevelLengthCategory, UUU) -> bool,
        InsertFn: FnMut(&'stored HSTreeNode, SegmentRange<UUU>, SegmentRange<UUU>),
    {
        let level = &self.level;
        for index in 0..level.nodes.len() as UUU {
            let candidate_range: Range<UUU> = level
                .candidate_range(self.query, self.threshold, index)
                .into();
            let len = level.substring_len(index);
            let len_category = self.coords.categorize_len(len);

            if cond(len_category, segment_index) {
                let query_index = self.segments.at(len_category, segment_index);
                // filter matches by position
                if candidate_range.contains(&query_index) {
                    let subquery_range = query_index..query_index + len;
                    let node = &self.level.nodes[index as usize];
                    let stored_substring_range = level.substring_range(index);
                    insert(node, subquery_range.into(), stored_substring_range);
                }
            }
        }
    }
}

impl<'stored> HSLengthGroup<'stored, UUU> {
    /// Returns the index of the deepest level
    fn height(&self) -> UUU {
        // it should be impossible to have a length group with 0 length from the public API
        debug_assert_ne!(self.levels.len(), 0);
        self.levels.len() as UUU - 1
    }
    /// Returns a length group with correctly sized levels
    fn new(length: UUU) -> HSLengthGroup<'stored, UUU> {
        let mut levels = Levels::<'stored>::default();
        let height = (length as f32).log2().floor() as usize;
        // hold enough levels for all substring lengths
        levels.resize(height + 1, Default::default());
        for depth in 0..height {
            let level = &mut levels[depth];
            // double the number of nodes in each level, start with 1 node
            level.nodes.resize(1 << depth, Default::default());
        }
        HSLengthGroup {
            levels,
            original: Default::default(),
            length,
        }
    }
    /// Populates all substring nodes for the string
    fn insert(&mut self, string: &'stored str) {
        if !self.original.contains(string) {
            self.original.insert(string);
            let source = self.original.get(string).unwrap();
            self.insert_rec(0, 0, string, source);
        }
    }
    /// Splits a string in half, with the left half having at most the length of the right
    fn split(string: &str) -> (&str, &str) {
        let len = string.len();
        let mid = len / 2;
        // the length of left is mid - 1 because mid is an exclusive bound
        let left = &string[0..mid];
        let right = &string[mid..len];
        (left, right)
    }
    /// Populates substring nodes downwards beginning from the node at depth and sibling_index
    fn insert_rec(
        &mut self,
        depth: usize,
        sibling_index: usize,
        string: &'stored str,
        original: &'stored str,
    ) {
        // no children at terminal depth
        let terminal = self.levels.len() - 1;
        if depth < terminal {
            // sibling index of next level, previous nodes all have 2 children
            let start = 2 * sibling_index;
            let (left, right) = Self::split(string);
            // add the left half and the right half to the next level
            self.insert_rec(depth + 1, start, left, original);
            self.insert_rec(depth + 1, start + 1, right, original);
        }
        let level = &mut self.levels[depth]; // this line is strangely order-sensitive
                                             // update the frequency for the substring
        level
            .frequencies
            .entry(string.clone())
            .and_modify(|freq| *freq += 1)
            .or_insert(0);
        let node = &mut level.nodes[sibling_index];
        // map the substring to the full string
        node.insert(string, original);
    }
    /// Returns map from frequencies to match information for strings
    /// whose substrings at the depth match at least `minimum_matches` candidate substrings in the query
    fn matches(&'stored self, query: &str, depth: UUU, minimum_matches: UUU) -> FrequencyMap<UUU> {
        // Following the CPMerge algorithm
        let level: &HSLevel<UUU> = &self.levels[depth as usize];
        // line 1 of CPMerge
        let segments: LevelSegments<UUU> =
            LevelSegments::<UUU>::new(query, depth, self.length, &level.frequencies);

        let level_coords = LevelCoordinates {
            depth,
            length: self.length,
        };
        let lesser_len = level_coords.lesser_len();
        // number of possible substrings of the lesser length from the query
        let n_lesser_substrings = query.len() as UUU - lesser_len + 1;

        // exclusive upper bound for iteration
        let lesser_end = n_lesser_substrings - minimum_matches + 1;
        // 1 less segment available for longer query substrings
        let greater_end = lesser_end - 1;

        let mut matches = StringSegmentMatches::<'stored, UUU>::default();

        let mut match_finder = MatchFinder::<'stored, '_> {
            coords: level_coords,
            level,
            query,
            threshold: minimum_matches,
            segments,
        };
        // lines 3-7 of CPMerge (segment_index is k)
        // guarantee at least 1 match for any string that meets the threshold (Signature)
        for segment_index in 0..lesser_end {
            match_finder.crawl(
                segment_index,
                // don't consider any longer substring if we ran out of signature indices
                // this is a way to avoid two loops with 0..lesser_end and 0..greater_end each only iterating through the lesser or greater substrings
                |len_category, segment_index| {
                    len_category != Greater || segment_index < greater_end
                },
                // add all matches from a node for a substring of the query as the index ranges
                |node, subquery_range, substring_range| {
                    let slice_range: Range<usize> = subquery_range.clone().into();
                    let subquery: &str = &query[slice_range];
                    if let Some(strings) = node.index.get(subquery) {
                        for &string in strings {
                            // because a BTreeSet is used in SegmentMatches, CPMerge shouldn't be weakened from visiting strings
                            // that match the same part of the subquery from multiple nodes
                            matches.insert(string, subquery_range.clone(), substring_range.clone());
                        }
                    }
                },
            );
        }
        let remaining_range = (lesser_end - 1)..query.len() as UUU;
        for segment_index in remaining_range {
            // lines 10-13 of CPMerge (but uses hash table lookup instead of binary search, and the for-loop is distributed)
            match_finder.crawl(
                segment_index,
                // don't consider shorter substring if we already considered it
                // this is a way to avoid having two loops with different lower bounds and only using lesser/greater substrings each
                |len_category, segment_index| len_category != Lesser || segment_index >= lesser_end,
                // merge the segments from inverted index in the node for each matching superstring if applicable
                |node, subquery_range, substring_range| {
                    matches.merge(node, subquery_range, query, substring_range);
                },
            );
            // lines 14-16 of CPMerge are omitted because the match frequency is required in another part of the algorithm

            // lines 17-19 of CPMerge
            // suppose there has only been one match for a stored string so far; then each remaining substring must correspond to at least one
            // match to meet the threshold
            let partial_minimum_matches = minimum_matches + segment_index + 1 - query.len() as UUU;
            matches.filter(partial_minimum_matches);
        }
        matches.to_sorted()
    }

    /// Returns an Option which contains the disjoint matches between `string` and `query` at the deepest level if there are enough based on `threshold`
    /// Otherwise, returns None at the first level after `previous_depth` where there are not enough matches to not exceed the `threshold`
    /// `matching_segments`: segment ranges in the query that match segments in `string`
    /// `threshold`: worst edit distance allowable (inclusive)
    fn greedy_match(
        &self,
        string: &str,
        query: &str,
        previous_depth: UUU,
        threshold: UUU,
        mut matching_segments: SegmentMatches<UUU>,
    ) -> Option<SegmentMatches<UUU>> {
        let mut paired_matching_segments: BTreeMap<SegmentRange<UUU>, SegmentRange<UUU>>;
        // Implement GREEDYMATCH from the paper
        for depth in (previous_depth + 1)..self.height() {
            let mut current_segments = SegmentMatches::default();

            let segments: UUU = 1 << depth;
            // bit vector, where each bit corresponds to whether the segment is from a matching segment in the previous level
            let mut trickled_segments = BitVec::<usize, Lsb0>::repeat(false, segments as usize);

            let level: &HSLevel<UUU> = &self.levels[depth as usize];
            // the last index into the previous level for a match from the previous level
            let mut last_sibling_index = 0;

            // lines 3-4, subsegments of matches also match
            for (subquery_segment, substring_segment) in matching_segments {
                let (subquery_left, subquery_right) = subquery_segment.split();
                let (substring_left, substring_right) = substring_segment.split();
                current_segments.insert(subquery_left, substring_left);
                current_segments.insert(subquery_right, substring_right);

                let substring_segment: Range<UUU> = substring_segment.into();
                // index of the stored string segment in the previous level
                last_sibling_index = level.index(substring_segment.start, last_sibling_index);
                let sibling_index = 2 * last_sibling_index as usize;
                // mark the two segments in the current level as being matched from a parent
                trickled_segments[sibling_index..=sibling_index + 1].fill(true);

                // the next index for a match from the last level will be at least 1 ahead
                last_sibling_index += 1;
            }
            // lines 5-9, add missing matches
            for index in 0..segments {
                let candidate_range: Range<usize> =
                    level.candidate_range(query, threshold, index).into();
                let substring_range: Range<usize> = level.substring_range(index).into();
                let substring = &string[substring_range];

                let matching_parent: bool = trickled_segments[index as usize];
                if !matching_parent {
                    if let Some(start) = query[candidate_range].find(substring) {
                        let start = start as UUU;
                        let subquery_range = start..start + level.substring_len(index);
                        current_segments.insert(subquery_range.into(), substring_range.into());
                    }
                }
            }
            matching_segments = StringSegmentMatches::<UUU>::maximum_disjoint_set(current_segments);

            // lines 10-11, prune if the edit distance is definitely more than `threshold`
            let minimum_matches = segments - threshold;
            let matches = matching_segments.len() as UUU;
            if matches < minimum_matches {
                return None;
            }
        }
        // return value needs to be 1-to-1 pairs between matches of segments in both the query and stored string
        // for the MultiExtension edit distance verification

        Some(matching_segments)
    }
}

impl Rankings<UUU> {
    /// Returns an inclusive upper bound for the Levenshtein distance for a string to be ranked
    fn threshold(&self) -> UUU {
        // we don't have enough results yet to lower our limit for the Levenshtein distance
        if self.best.len() < self.limit as usize {
            self.initial_threshold
        }
        // we won't accept any distance that isn't better than the worst so far in the top-k
        else {
            // this will fail if best.len() is 0 and limit is also 0, which shouldn't be possible from the public API
            let worst = self.best.peek().unwrap();
            worst.distance - 1
        }
    }
    /// Returns Rankings able to contain up to `limit` rankings and defaulting to `initial_threshold`
    /// with less than `limit` rankings
    fn new(limit: UUU, initial_threshold: UUU) -> Rankings<UUU> {
        Rankings {
            best: Default::default(),
            limit,
            initial_threshold,
        }
    }
    /// Updates the rankings with `string` and `distance`, keeping at most `self.limit` rankings
    fn rank(&mut self, string: &str, distance: UUU) {
        let string = string.to_string();
        self.best.push(MeasuredString::<UUU> { string, distance });
        if (self.best.len() > self.limit as usize) {
            self.best.pop();
        }
    }
    /// Returns all rankings as strings with their Levenshtein distance in ascending order over edit distance
    pub fn into_measured_strings(self) -> Vec<MeasuredString<UUU>> {
        self.best.into_sorted_vec()
    }
    /// Returns all ranked strings sorted by ascending Levenshtein distance
    pub fn into_strings(self) -> Vec<String> {
        self.into_measured_strings()
            .into_iter()
            .map(|measured| measured.string)
            .collect()
    }
}

impl LengthSelector<UUU> {
    /// Returns a LengthSelector given the length of the query and the current threshold distance
    fn new(query_len: UUU, threshold: UUU, max_len: UUU) -> Self {
        // inclusive bounds for possible lengths to check
        // prevent underflow by saturating at 0
        let lower_bound = query_len.saturating_sub(threshold);
        let upper_bound = min(max_len, query_len + threshold);

        let lower_range = lower_bound..query_len;
        let upper_range = query_len..(upper_bound + 1);

        LengthSelector {
            lower_range,
            upper_range,
            query_len,
        }
    }
}

impl Iterator for LengthSelector<UUU> {
    type Item = UUU;
    /// Returns the next length to search that's closest to the query length
    fn next(&mut self) -> Option<UUU> {
        // under/overflow errors shouldn't be possible here because of how the ranges are initialized
        let lower_length_dist = self.query_len - self.lower_range.end + 1;
        let upper_length_dist = self.upper_range.start - self.query_len;

        // the next closest length is from the lower range
        if lower_length_dist < upper_length_dist && !self.lower_range.is_empty() {
            self.lower_range.end -= 1;
            Some(self.lower_range.end)
        } else if !self.upper_range.is_empty() {
            let next_len = self.upper_range.start;
            self.upper_range.start += 1;
            Some(next_len)
        } else {
            None
        }
    }
}

impl<'stored> HSTree<'stored, UUU> {
    /// Inserts a string into the tree
    pub fn insert(&mut self, string: &'stored str) {
        let length = string.len() as UUU;
        if length > 0 {
            self.groups
                .entry(length)
                .and_modify(|group| group.insert(string))
                .or_insert({
                    let mut group = HSLengthGroup::<UUU>::new(length);
                    group.insert(string);
                    group
                });
        }
    }
    /// Returns up to `limit` strings with the closest Levenshtein distance to `query`,
    /// The result is sorted by ascending Levenshtein distance
    pub fn top(&self, query: &str, limit: UUU) -> Result<Rankings<UUU>, &str> {
        if self.groups.is_empty() {
            return Err("No data stored in the search index");
        }
        if query.is_empty() {
            return Ok(Default::default());
        }

        // lazy way of handling query larger than UUU (truncate)
        // query is unlikely to be larger than UUU for typical use cases
        // TODO?: better support for queries larger than UUU (add yet another type parameter to bound query size?)
        let mut query = query;
        if query.len() > UUU::MAX as usize { query = &query[0..UUU::MAX as usize]; }
        let query = query;

        // extreme lengths of strings currently in the tree
        let min_len = *self.groups.first_key_value().unwrap().0;
        let max_len = *self.groups.last_key_value().unwrap().0;
        let query_len = query.len() as UUU;
        // inclusive upper bound on edit distance allowed so far (avoid searching lengths less than the minimum or more than the maximum stored)
        // this is equal to UB_q - 1 from the paper
        let mut inclusive_threshold = max(query_len.abs_diff(min_len), query_len.abs_diff(max_len));
        // closure that returns the minimum number of matches needed to have an edit distance of at most `inclusive_threshold`
        let minimum_matches = |segments: UUU| segments.saturating_sub(inclusive_threshold);

        let mut rankings = Rankings::<UUU>::new(limit, inclusive_threshold);
        // the deepest level of any group
        let max_level = (max_len as f32).log2().floor() as UUU;

        // I'm questioning why HS-Topk iterates through length groups per level, rather than levels per length group
        // I think it's because the loop termination condition changes bounds on the match groups in the frequency map,
        // so revisiting a level in another length group could result in some strings being missed
        for depth in 1..=max_level {
            // number of nodes in previous level
            let previous_segments = 1 << (depth - 1);
            if previous_segments > inclusive_threshold + 1 {
                break;
            }
            let mut length_selector =
                LengthSelector::<UUU>::new(query_len, inclusive_threshold, max_len);

            // this could be more efficient using the Cursor API of BTreeMap (from lower_bound(), upper_bound())
            // to avoid checking non-existent lengths, which would also make using a BTreeMap more advantageous than a HashMap
            // however, the Cursor API is currently nightly-only as of Rust 1.74.1
            while let Some(length) = length_selector.next() {
                if let Some(group) = self.groups.get(&length) {
                    // TODO?: use a new map from length to &LengthGroup to remove short groups on the fly
                    if group.height() >= depth {
                        // line 6 of HS-Topk
                        let segments = previous_segments << 1;
                        let frequency_map = group.matches(query, depth, minimum_matches(segments));

                        for (frequency, matches) in frequency_map.into_iter().rev() {
                            let matches = matches.matches;
                            // stop if the edit distance threshold lowered and cut off groups of smaller matches starting from here
                            if matches.len() < minimum_matches(segments) as usize {
                                break;
                            }
                            for (string, segments) in matches {
                                if let Some(matches) = group.greedy_match(
                                    string,
                                    query,
                                    depth,
                                    inclusive_threshold,
                                    segments,
                                ) {
                                    // lines 10-12 of HS-Topk
                                    if let Some(distance) = Self::multi_extension(
                                        string,
                                        query,
                                        inclusive_threshold,
                                        matches,
                                        // the group must have levels
                                        group.levels.last().unwrap(),
                                    ) {
                                        rankings.rank(string, distance);
                                        inclusive_threshold = rankings.threshold();
                                        // this is used to change the bounds on which lengths to visit before finishing the current level
                                        length_selector = LengthSelector::<UUU>::new(
                                            query_len,
                                            inclusive_threshold,
                                            max_len,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(rankings)
    }
    /// Returns an Option with the edit distance between `query_substring` and `substring` if it is at most the `threshold`, otherwise returns None
    fn length_aware(query_substring: &str, substring: &str, threshold: UUU) -> Option<UUU> {
        // Implementing the Length-aware verification from doi:10.14778/2078331.2078340
        let (query_len, substring_len) = (query_substring.len(), substring.len());
        let delta = query_len.abs_diff(substring_len) as UUU;
        if delta > threshold {
            return None;
        }
        let left = ((threshold - delta) / 2) as usize;
        // watch out for overflow (there may still be an edge case for UUU = u64 but I don't think the overhead of u128 is worth it)
        let right = ((threshold as u64 + delta as u64) / 2) as usize;
        let max_row_search_length = right - left + 1;

        // using a two-row memoization https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_two_matrix_rows
        // adapted for Length-aware with shortened rows
        let distance_rows = Array2::<UUU>::zeros((2, max_row_search_length));
        let prev_row = &mut distance_rows.row(0);
        let current_row = &mut distance_rows.row(1);

        for j in 0..=right {
            prev_row[j] = j as UUU;
        }
        let mut prev_max_row_index = right;

        for i in 1..=query_len as usize {
            // if the edit distances for suffixes in the row checked so far must all be greater than `threshold`
            let mut all_bad = true;
            // edit_matrix[[i, 0]] has been computed where necessary
            let lower = max(1, i.saturating_sub(left));
            let upper = min(i.saturating_add(right), substring_len);
            let query_suffix_len = query_len - i;
            for j in lower..=upper {
                let substring_suffix_len = substring_len - j;

                // compare the graphemes at i - 1 and j - 1
                // (edit_matrix[[0 , 0]] is the edit distance between two empty strings, so i and j are offset by 1)
                let query_grapheme = &query_substring[i - 1..i];
                let substring_grapheme = &substring[j - 1..j];
                let diff = (query_grapheme != substring_grapheme) as u8;

                // convert matrix index to two-row index
                let j = j - lower;
                // the current row is staggered right by 1, so the index into the previous row for substitution is j
                let replace_dist = prev_row[j] + diff;
                let mut dist = replace_dist;

                // some rows are only partially filled
                if j < prev_max_row_index {
                    let insert_dist = prev_row[j + 1] + 1;
                    dist = min(dist, insert_dist);
                }
                if j > 0 {
                    let erase_dist = current_row[j - 1] + 1;
                    dist = min(dist, erase_dist);
                }

                current_row[j] = dist;

                // Lemma 4 from the paper
                let expected_edit_distance =
                    dist as usize + query_suffix_len.abs_diff(substring_suffix_len);
                all_bad &= expected_edit_distance > threshold as usize;
            }
            if all_bad {
                return None;
            }
            std::mem::swap(prev_row, current_row);
            prev_max_row_index = upper - lower;
        }
        Some(current_row[prev_max_row_index])
    }

    /// Returns an Option with the edit distance between `string` and `query` if it is at most the `threshold`, otherwise returns None
    fn multi_extension(
        string: &str,
        query: &str,
        threshold: UUU,
        matches: BTreeMap<SegmentRange<UUU>, SegmentRange<UUU>>,
        level: &HSLevel<UUU>,
    ) -> Option<UUU> {
        let mut prev_match_index = 0;
        unimplemented!()
        /*
        let mut match_index = level.index(prev_match_index, )
        "foo"
        */
    }
}
/*
        )*
    }
}
*/

//impl_hs_tree!(u8 u16 u32 u64);
