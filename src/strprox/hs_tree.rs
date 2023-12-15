use crate::strprox::gats::*;
use std::cmp::Ordering;
use std::cmp::{max, min};
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

/// Contains a map from substrings to original strings
/// The substrings are restricted to a certain length and position
#[derive(Clone, Default)]
pub struct HSTreeNode<'a, MapTypeConstructor: MapGAT = HashMapGAT> {
    index: MapTypeConstructor::Map<&'a str, &'a str>,
}

impl<MapTypeConstructor: MapGAT> HSTreeNode<'_, MapTypeConstructor> {
    /// Returns length of substrings in the node
    fn len(&self) -> u32 {
        match self.index.iter().next() {
            // Invariant: all substrings should have the same length
            Some((string, _)) => string.len() as u32,
            None => 0,
        }
    }
}

pub struct HSLevel<'a> {
    // this should be at most the length of the original string, so iteration shouldn't be prohibitive
    nodes: Vec<HSTreeNode<'a>>,
    // stores frequencies of substrings in this level (for CPMerge)
    frequencies: HashMap<&'a str, u32>,
}

struct LevelCoordinates {
    depth: u32,
    /// Length of original string
    length: u32,
}

/// enum for whether a node has the smaller or larger length in a level
enum LevelLengthCategory {
    Lesser = 0,
    Greater = 1,
}
use LevelLengthCategory::{Greater, Lesser};

impl LevelCoordinates {
    /// Returns the minimum possible length of a substring in this level
    fn lesser_len(&self) -> u32 {
        self.depth / (1 << self.length)
    }
    /// Returns the maximum possible length of a substring in this level
    fn greater_len(&self) -> u32 {
        self.lesser_len() + 1
    }
    /// Returns the minimum and maximum lengths of substrings in this level
    fn lengths(&self) -> (u32, u32) {
        let lesser = self.lesser_len();
        (lesser, lesser + 1)
    }
    /// Categorizes the length as Lesser or Greater (Greater by default)
    fn categorize_len(&self, len: u32) -> LevelLengthCategory {
        if len == self.lesser_len() {
            Lesser
        } else {
            Greater
        }
    }
}

/// Represents positional information of a substring in a certain node
struct TreeCoordinates {
    depth: u32,
    sibling_index: u32,
    /// Length of original string
    length: u32,
}

impl TreeCoordinates {
    /// Computes the starting index of a substring in its original string at the tree coordinates
    fn pos(&self) -> u32 {
        ((self.sibling_index as u64 * self.length as u64) / (1 << self.depth)) as u32
    }
    /// Computes length of substrings at the tree coordinates
    fn len(&self) -> u32 {
        let size = 1 << self.depth;
        let sibling_index = self.sibling_index as u64;
        let length = self.length as u64;
        // pos of next node - pos of current node
        (((sibling_index + 1) * length) / size - (sibling_index * length) / size) as u32
    }
    /// Computes range of indices to query segments that may match a substring in a node
    fn candidate_range(&self, query: &str, threshold: u32) -> Range<u32> {
        let pos = self.pos();
        let len = self.len();
        let delta = query.len() as u32 - len;
        // pos and sibling_index begin at 0, unlike in the paper
        let lower = max(
            0,
            max(
                pos - self.sibling_index,
                pos + delta - threshold + self.sibling_index,
            ),
        );
        let upper = min(
            delta,
            min(
                pos + self.sibling_index,
                pos + delta + threshold - self.sibling_index,
            ),
        );
        lower..(upper + 1)
    }
    fn len_category(&self) -> LevelLengthCategory {
        // the lesser length in the level
        let min_len = LevelCoordinates {
            depth: self.depth,
            length: self.length,
        }
        .lesser_len();
        let len = self.len();
        if len == min_len {
            Lesser
        } else {
            Greater
        }
    }
}

impl From<TreeCoordinates> for LevelCoordinates {
    fn from(value: TreeCoordinates) -> Self {
        LevelCoordinates {
            depth: value.depth,
            length: value.length,
        }
    }
}

impl From<LevelCoordinates> for TreeCoordinates {
    fn from(value: LevelCoordinates) -> Self {
        TreeCoordinates {
            depth: value.depth,
            sibling_index: 0,
            length: value.length,
        }
    }
}

pub struct HSLengthGroup<'a> {
    // avoid explicit branch representation to allow for efficient traversal
    levels: Vec<HSLevel<'a>>,
    // the full strings inside the group
    original: HashSet<&'a str>,
    length: u32,
}

/// Represents the segmentation of a query string to compare with the substrings at a certain tree level
struct LevelSegments<'a> {
    /// Starting indices of each segment sorted in order of match frequency
    ///
    /// A level in the tree has substrings of two possible lengths
    /// The first vector contains the indices for the shorter segments
    /// The second vector contains the indices for the longer segments
    segments: [Vec<u32>; 2],
    query: &'a str,
}

impl LevelSegments<'_> {
    /// Returns vector of frequencies of each query segment
    /// `segments`: number of segments
    /// `len`: length of each segment
    fn frequencies(
        segments: u32,
        len: u32,
        query: &str,
        frequencies: &HashMap<&str, u32>,
    ) -> Vec<u32> {
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
    fn new<'a>(
        query: &'a str,
        depth: u32,
        length: u32,
        frequencies: &HashMap<&str, u32>,
    ) -> LevelSegments<'a> {
        // minimum length of a substring at the level given by depth and length in the tree
        let min_len = length / (1 << depth);
        let mut segments: [Vec<u32>; 2] = Default::default();
        // number of shorter substring segments
        let n_shorter = query.len() as u32 - min_len + 1;
        // length of substrings in a level can differ by at most 1
        let n_longer = n_shorter - 1;
        segments[0] = (0..n_shorter).collect();
        // longer substring segments
        segments[1] = (0..query.len() as u32 - min_len).collect();

        let mut segment_frequencies: [Vec<u32>; 2] = Default::default();
        segment_frequencies[0] = LevelSegments::frequencies(n_shorter, min_len, query, frequencies);
        segment_frequencies[1] =
            LevelSegments::frequencies(n_longer, min_len + 1, query, frequencies);
        for i in 0..2 {
            let segment_frequencies = segment_frequencies[i];
            // sort the segment starting positions from those with least associated substrings to those with the most
            // for CPMerge
            segments[i].sort_by(|a, b| {
                segment_frequencies[*a as usize].cmp(&segment_frequencies[*b as usize])
            });
        }

        LevelSegments { segments, query }
    }
    fn at(&self, len_category: LevelLengthCategory, index: u32) -> u32 {
        self.segments[len_category as usize][index as usize]
    }
}

#[derive(PartialEq, Eq)]
struct SegmentRange {
    range: Range<u32>,
}

impl PartialOrd for SegmentRange {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SegmentRange {
    fn cmp(&self, other: &Self) -> Ordering {
        // not using range as Iterator's cmp because it would iterate through the range
        let start_order = self.range.start.cmp(&other.range.start);
        if start_order != Ordering::Equal {
            return start_order;
        }
        self.range.end.cmp(&other.range.end)
    }
}

impl From<Range<u32>> for SegmentRange {
    fn from(value: Range<u32>) -> Self {
        SegmentRange { range: value }
    }
}

impl From<SegmentRange> for Range<u32> {
    fn from(value: SegmentRange) -> Self {
        value.range
    }
}

#[derive(Default)]
struct SegmentMatches<'a> {
    // matching query substring ranges for original strings of each matched substring in the tree
    matches: HashMap<&'a str, BTreeSet<SegmentRange>>,
}

impl<'a> SegmentMatches<'a> {
    /// Returns the maximum number of disjoint segments that were matched between the query and `string`,
    /// consuming the match information
    /// Replaces SEGCOUNT from the paper from doi: 10.1109/ICDE.2015.7113311
    fn maximum_disjoint_set_size(&mut self, string: &str) -> u32 {
        // https://en.wikipedia.org/wiki/Maximum_disjoint_set#1-dimensional_intervals:_exact_polynomial_algorithm
        // This will result in the cardinality of the maximum disjoint set, so dynamic programming is unnecessary
        let mut size: u32 = 0;
        if let Some(ranges) = self.matches.get(string) {
            let mut upper: i32 = -1;
            for range in ranges {
                let range: Range<u32> = (*range).into();
                // skip all overlapping ranges (the first range with the end at upper cannot intersect more than the others)
                if range.start as i32 > upper {
                    size += 1;
                    upper = range.end as i32;
                }
            }
            self.matches.remove(string);
        }
        size
    }

    /// Inserts a matching substring range in the query for the string
    fn insert(&mut self, string: &str, subquery_range: Range<u32>) {
        if let Some(mut ranges) = self.matches.get_mut(&string) {
            ranges.insert(subquery_range.into());
        } else {
            let mut ranges = BTreeSet::<SegmentRange>::new();
            ranges.insert(subquery_range.into());
            self.matches.insert(string, ranges);
        }
    }
    /// Removes all matches for the string
    fn remove(&mut self, string: &str) {
        self.matches.remove(string);
    }
    /// Returns number of matching substrings for the string
    fn count(&self, string: &str) -> u32 {
        if let Some(ranges) = self.matches.get(string) {
            ranges.len() as u32
        } else {
            0
        }
    }
    /// Consumes the matches and returns a map from match frequency to original string
    fn to_sorted(self) -> BTreeMap<u32, HashSet<&'a str>> {
        let mut frequency_map = BTreeMap::<u32, HashSet<&str>>::new();
        for (string, ranges) in self.matches {
            let frequency = ranges.len() as u32;
            let mut mapped: &mut HashSet<&str>;
            if let Some(strings) = frequency_map.get_mut(&frequency) {
                mapped = strings;
            } else {
                frequency_map.insert(frequency, Default::default());
                mapped = frequency_map.get_mut(&frequency).unwrap();
            }
            mapped.insert(string);
        }
        frequency_map
    }
}

struct MatchFinder<'a> {
    coords: LevelCoordinates,
    level: &'a HSLevel<'a>,
    query: &'a str,
    threshold: u32,
    segments: &'a LevelSegments<'a>,
    matches: &'a mut SegmentMatches<'a>,
}

impl MatchFinder<'_> {
    /// Adds substring matches within a level
    fn crawl(
        &mut self,
        segment_index: u32,
        cond: fn(LevelLengthCategory, u32) -> bool,
        insert: fn(&mut SegmentMatches, &HSTreeNode),
    ) {
        let mut coords = TreeCoordinates::from(self.coords);

        while coords.sibling_index < self.level.nodes.len() as u32 {
            let candidate_range = coords.candidate_range(self.query, self.threshold);
            let len = coords.len();
            let len_category = LevelCoordinates::from(coords).categorize_len(len);
            if cond(len_category, segment_index) {
                let query_index = self.segments.at(len_category, segment_index);
                if candidate_range.contains(&query_index) {
                    let query_index = query_index as usize;
                    let subquery_range = query_index..query_index + len as usize;
                    let subquery = &self.query[subquery_range];
                    let node: &HSTreeNode = &self.level.nodes[coords.sibling_index as usize];
                    insert(self.matches, node);
                }
            }
            coords.sibling_index += 1;
        }
    }
}

impl HSLengthGroup<'_> {
    /// Create a length group with correctly sized levels
    fn new(length: u32) -> HSLengthGroup {
        let mut group: HSLengthGroup;
        let height = floor(log2(length));
        // hold enough levels for all substring lengths
        group.levels.resize(height, Default::default());
        for depth in 0..height {
            let level = &group.levels[depth];
            // double the number of nodes in each level, start with 1 node
            level.nodes.resize(1 << depth, Default::default());
        }
        group
    }
    /// Populates all substring nodes for the string
    fn insert(&mut self, string: String) {
        if !self.data.contains(&string) {
            self.data.insert(string.clone());
            let source = self.data.get(&string).unwrap();
            let height = self.levels.len();
            self.insert_rec(0, sibling_index, string, source);
        }
    }
    /// Populates substring nodes downwards beginning from the node at depth and sibling_index
    fn insert_rec(&mut self, depth: u32, sibling_index: u32, string: String, original: &str) {
        let level: &HSLevel = &self.levels[depth];
        // no children at terminal depth
        let terminal = self.levels.len() - 1;
        if depth < terminal {
            let len = string.len();
            let mid = len / 2;
            let left = string[0..mid].to_string();
            let right = string[mid..len].to_string();
            // sibling index of next level, previous nodes all have 2 children
            let start = 2 * sibling_index;
            // add the left half and the right half to the next level
            self.insert_rec(depth + 1, start, left, original);
            self.insert_rec(depth + 1, start + 1, right, original);
        }
        // update the frequency for the substring
        level
            .frequencies
            .entry(string.clone())
            .and_modify(|freq| freq += 1)
            .or_insert(0);
        let mut node = &level.nodes[sibling_index];
        // map the substring to the full string
        node.index.insert(string, original);
    }
    /// Returns strings whose substrings at the depth match at least `threshold` candidate substrings in the query
    fn matches(&self, query: &str, depth: u32, threshold: u32) -> BTreeMap<u32, HashSet<&str>> {
        let result = BTreeMap::<u32, HashSet<&str>>::new();
        let level: &HSLevel = &self.levels[depth];
        let segments = LevelSegments::new(query, depth, self.length, &level.frequencies);
        let mut matches: SegmentMatches = Default::default();

        let level_coords = LevelCoordinates {
            depth,
            length: self.length,
        };
        let lesser_len = level_coords.lesser_len();
        let lesser_end: u32 = (query.len() - lesser_len) - threshold + 1;
        // 1 less segment available for longer query substrings
        let greater_end = lesser_end - 1;

        let match_finder = MatchFinder {
            coords: level_coords,
            level,
            query,
            threshold,
            segments: &segments,
            matches: &mut matches,
        };
        // index into the vector of query segment indices for the larger possible substrings in this level
        // includes the range for shorter substrings 0..lesser_end
        // guarantee at least 1 match for any string that meets the threshold (Signature)
        for segment_index in 0..lesser_end {
            match_finder.crawl(
                segment_index,
                // don't consider any longer substring if we ran out of signature indices
                |len_category, segment_index| {
                    len_category != Greater || segment_index < greater_end
                },
                |matches, node| {
                    if let Some(strings) = node.index.get(subquery) {
                        for string in strings {
                            matches.insert(string, subquery_range);
                        }
                    }
                },
            );
        }
        let remaining_range = (lesser_end - 1)..(query.len() - 1);
        for segment_index in remaining_range {
            match_finder.crawl(
                segment_index,
                // don't consider shorter substring if we already considered it
                |len_category, segment_index| len_category != Lesser || segment_index >= lesser_end,
                |matches, node| {
                    for string in matches.matches.keys() {
                        if node.index.contains_key(string) {
                            matches.insert(string, subquery_range);
                        }
                    }
                },
            );
            // CPMerge
            for (string, ranges) in &matches.matches {
                if ranges.len() + query.len() - segment_index - 1 < threshold {
                    matches.remove(string);
                }
            }
        }
        matches.to_sorted()
    }
}

/// Structure that associates a string with its Levenshtein distance from the query
#[derive(PartialEq, Eq, PartialOrd)]
pub struct MeasuredString {
    pub string: String,
    pub distance: u32,
}

impl Ord for MeasuredString {
    /// Compare the edit distances for MeasuredString
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}

/// Structure that holds at most `limit` of the strings closest to the query
pub struct Rankings {
    best: BinaryHeap<MeasuredString>,
    limit: u32,
}

impl Rankings {
    /// Returns a threshold for the Levenshtein distance for a string to be placed into the best results
    fn threshold(&self) -> u32 {
        // we don't have enough results yet to limit our Levenshtein distance
        if self.best.len() < self.limit as usize {
            u32::MAX
        }
        // we won't accept any distance worse than the worst so far in the top-k
        else {
            // this will fail if best.len() is 0 and limit is also 0
            let worst = self.best.peek().unwrap();
            worst.distance
        }
    }
    fn new(limit: u32) -> Rankings {
        Rankings {
            best: Default::default(),
            limit,
        }
    }
}

#[derive(Default)]
pub struct HSTree {
    // indexed by length, then depth, then sibling number
    groups: BTreeMap<u32, HSLengthGroup>,
}

impl HSTree {
    /// Inserts a string into the tree
    fn insert(&mut self, string: String) {
        let length = string.len() as u32;
        groups
            .entry(length)
            .and_modify(|group| group.insert(string))
            .or_insert({
                let group = HSLengthGroup::new(length);
                group.insert(string);
                group
            });
    }
    /// Returns up to `limit` strings with the closest Levenshtein distance to `query`,
    /// The result is sorted by ascending Levenshtein distance
    pub fn top(&self, query: &str, limit: u32) -> Vec<MeasuredString> {
        let mut best = Rankings::new(limit);
        let mut upper_bound = u32::MAX;
        for i in 0..query.len() {
            if (1 << i) >= (upper_bound + 1) {
                break;
            }
            // saturating sub/add prevents under/overflow
            for length in (query.len().saturating_sub(upper_bound))
                ..=(query.len().saturating_add(upper_bound))
            {
                if let Some(group) = &self.tree.groups.get(length) {}
            }
        }
        best.into_sorted_vec()
    }
}
