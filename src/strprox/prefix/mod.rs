use std::{
    cmp::min,
    collections::{BTreeSet, HashMap},
    ops::Range,
    string,
};
use priority_queue::PriorityQueue;

use super::{MeasuredPrefix, MeasuredString};

/// Implements "Depth-Based Fuzzy Auto-Completion" (DFA) from https://doi.org/10.1016/j.aej.2020.06.012
///
/// Also references the baseline algorithm
/// "Matching-Based Method for Error-Tolerant Autocompletion" (META) from https://doi.org/10.14778/2977797.2977808

type UUU = u8;

/// A trie node with a similar structure from META
#[derive(PartialEq, Eq, Hash, Debug)]
struct Node<UUU> {
    /// One Unicode character
    character: char,
    /// Range of indices into descendant nodes
    descendant_range: Range<usize>,
    /// Range of indices into strings with the prefix from this node
    string_range: Range<usize>,
    /// Length of the prefix from this node
    // this is needed because the nodes are pushed onto a structure outside the trie, where their depths would otherwise be forgotten
    depth: UUU,
}

type TrieStrings<'stored> = Vec<&'stored str>;
type TrieNodes<UUU> = Vec<Node<UUU>>;

#[derive(Debug)]
pub struct Trie<'stored, UUU> {
    nodes: TrieNodes<UUU>,
    /// Stored strings
    strings: TrieStrings<'stored>,
}

/// Returns an Option with the next valid Unicode scalar value after `character`, unless `character` is char::MAX
#[inline]
fn char_succ(character: char) -> Option<char> {
    let mut char_range = character..=char::MAX;
    char_range.nth(1)
}

/// The "matching node" structure from DFA
// This struct could be more space-efficient if the query's length can be parametrically bounded below usize
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub(crate) struct MatchingNode<UUU> {
    /// Index into the trie's nodes
    node_id: usize,

    // 1-based index into the query string where the node's character matches
    // does not appear to actually be used in the algorithm, instead only for illustration
    //query_match_index: usize,
    /// The edit distance of the trie string prefix from the query prefix at the seen last match
    //prefix_edit_distance: UUU,
    /// The length of the query string prefix that has been checked by the node
    checked_len: usize,
    /// The edit distance of the trie string prefix from the query prefix
    edit_distance: UUU,
}

impl Ord for MatchingNode<UUU> {
    /// Order lower edit distances first, then higher query indices first (from DFA),
    /// then the `node_id` to allow insertion into a set when the other two are equal
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.edit_distance
            .cmp(&other.edit_distance)
            .then_with(|| other.checked_len.cmp(&self.checked_len))
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

impl PartialOrd for MatchingNode<UUU> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'stored> Trie<'stored, UUU> {
    /// Returns the root node of the trie (panics if the trie is empty)
    fn root(&self) -> &Node<UUU> {
        self.nodes.first().unwrap()
    }
    /// Returns trie over `source` (expects `source` to have at most usize::MAX - 1 strings)
    pub fn new(source: &'stored [&'stored str]) -> Self {
        let mut strings = TrieStrings::<'stored>::with_capacity(source.len());
        for string in source.iter() {
            strings.push(string);
        }
        // sort and dedup to compute the `string_range` for each node using binary search
        strings.sort();
        strings.dedup();

        // rough estimate on the size of the trie
        let nodes = TrieNodes::with_capacity(3 * source.len());

        let mut trie = Self { strings, nodes };

        // Construct all nodes
        trie.init_nodes(
            &mut 0,
            0,
            &mut Default::default(),
            '\0',
            0,
            0,
            trie.strings.len(),
        );
        trie
    }
    /// `last_char` is the last character in the prefix
    fn init_nodes(
        &mut self,
        node_id: &mut usize,
        depth: UUU,
        prefix: &mut String,
        last_char: char,
        suffix_start: usize,
        start: usize,
        end: usize,
    ) {
        let current_id = node_id.clone();

        let current_node = Node {
            character: last_char,
            // change the descendant range later
            descendant_range: Default::default(),
            string_range: start..end,
            depth,
        };
        // the current node is added before all the descendants,
        // and its location in `nodes` is `current_id`
        debug_assert_eq!(self.nodes.len(), current_id);
        self.nodes.push(current_node);

        // the next node, if it exists, will have 1 higher id
        *node_id += 1;

        // `node_id` is required to be incremented in pre-order to have continuous `descendant_range``
        let mut child_start = start;
        while child_start != end {
            // add to the prefix
            let suffix = &self.strings[child_start][suffix_start..];
            if let Some(next_char) = suffix.chars().next() {
                // strings in strings[child_start..child_end] will have the same prefix
                let child_end;
                let next_prefix;

                // get the boundary in `strings` for strings with the prefix extended with next_char
                if let Some(succ) = char_succ(next_char) {
                    // `lexicographic_marker` is the first string that's lexicographically ordered after all strings with prefix
                    let lexicographic_marker = &mut *prefix;
                    lexicographic_marker.push(succ);

                    // offset from start where the lexicographic marker would be
                    let offset;
                    match self.strings[start..end].binary_search(&lexicographic_marker.as_str()) {
                        // same bound either way, but if it's Err it will be the last iteration
                        Ok(x) => offset = x,
                        Err(x) => offset = x,
                    }
                    child_end = start + offset;

                    debug_assert!(child_end > child_start);

                    next_prefix = lexicographic_marker;
                    next_prefix.pop();
                } else {
                    // the next character in the prefix is char::MAX,
                    // so this must be the last prefix from the current one
                    debug_assert_eq!(next_char, char::MAX);
                    child_end = end;
                    next_prefix = prefix;
                }
                next_prefix.push(next_char);

                // requires nightly
                //let next_suffix_start = strings[start].ceil_char_boundary(suffix_start + 1);

                let next_suffix_start = suffix_start + next_char.len_utf8();

                // Construct all descendant nodes with the next prefix
                self.init_nodes(
                    node_id,
                    depth + 1,
                    next_prefix,
                    next_char,
                    next_suffix_start,
                    child_start,
                    child_end,
                );

                // reset the prefix state
                let prefix = next_prefix;
                prefix.pop();

                // look at strings with a different next character in their prefix
                child_start = child_end;
            } else {
                /*
                // Special leaf nodes are not required because of `string_range`
                let leaf_node = Node {
                    character: '\0',
                    descendant_range: 0..0,
                    string_range: child_start..child_start + 1,
                    depth: depth + 1,
                };

                debug_assert_eq!(self.nodes.len(), *node_id);
                self.nodes.push(leaf_node);
                *node_id += 1;
                */

                // this string has already been accounted for by the parent node,
                // whose prefix is equal to the whole string
                child_start += 1;
            }
        }

        // node_id is now 1 greater than the index of the last in-order node that's in the subtree from the current node
        let descendant_range = current_id + 1..*node_id;
        self.nodes[current_id].descendant_range = descendant_range;
    }
    /// Visits all nodes in pre-order, calling the `visitor` on each node
    fn preorder<VisitorFn>(&self, visitor: &mut VisitorFn)
    where
        VisitorFn: FnMut(&Node<UUU>),
    {
        self.preorder_node(visitor, self.root());
    }
    /// Visits `node` and all descendants in pre-order, calling the `visitor` on each
    fn preorder_node<VisitorFn>(&self, visitor: &mut VisitorFn, node: &Node<UUU>)
    where
        VisitorFn: FnMut(&Node<UUU>),
    {
        visitor(node);
        for node_id in node.descendant_range.clone() {
            let descendant = &self.nodes[node_id];
            self.preorder_node(visitor, descendant);
        }
    }
    /// Returns the top `requested` number of strings with the best prefix distance from the query,
    /// or all strings sorted by prefix edit distance if `requested` is larger than the number stored
    pub fn autocomplete(&self, query: &str, requested: usize) -> Vec<(String, MatchingNode<UUU>)> {
        if requested == 0 {
            return Default::default();
        }

        // Implementing the DFA algorithm
        // need to iterate through the characters of the query
        let query: Vec<char> = query.chars().collect();

        let mut result = HashMap::<&'stored str, MatchingNode<UUU>>::new();
        let mut priority_queue = PriorityQueue::<&Node<UUU>, MatchingNode<UUU>>::new();
        // using BTreeSet instead of BinaryHeap because removal by value is required
        // ideally, first() and pop_first() would be O(1)
        let mut priority_queue = BTreeSet::<MatchingNode<UUU>>::new();
        let matching_node = MatchingNode::<UUU> {
            node_id: 0, // the root is at index 0
            //query_match_index: 0,
            //prefix_edit_distance: 0,
            checked_len: 0,
            edit_distance: 0,
        };
        priority_queue.insert(matching_node);

        let mut map = HashMap::<&Node<UUU>, MatchingNode<UUU>>::new();

        while result.len() < requested {
            if let Some(mut matching_node) = priority_queue.pop_first() {
                if (matching_node.checked_len as usize) < query.len() {
                    self.check_descendants(
                        query.as_slice(),
                        &matching_node,
                        &mut priority_queue,
                        &mut map,
                    );
                    matching_node.checked_len += 1;
                    matching_node.edit_distance += 1;
                    priority_queue.insert(matching_node);
                } else {
                    self.get_matching_node_strings(&matching_node, &mut result, requested);
                }
            } else {
                // stop when the priority queue is empty, which is when there aren't enough strings stored according to the paper
                debug_assert!(self.strings.len() < requested);
                break;
            }
        }

        // transform the result to a vector
        let mut result: Vec<_> = result
            .into_iter()
            .map(|(string, matching_node)| (string.to_string(), matching_node))
            .collect();

        result.sort();
        result
    }

    /// Implements the CHECK-DESCENDANTS algorithm from DFA
    fn check_descendants(
        &'stored self,
        query: &[char],
        matching_node: &MatchingNode<UUU>,
        priority_queue: &mut BTreeSet<MatchingNode<UUU>>,
        map: &mut HashMap<&'stored Node<UUU>, MatchingNode<UUU>>,
    ) {
        let node = &self.nodes[matching_node.node_id];

        const ERROR: usize = 0;

        // there may be an edge case for a long query whose length is close to UUU::MAX
        // where UUU can overflow and cause incorrect results without being cast to usize
        let depth_threshold = node.depth as usize + matching_node.edit_distance as usize + 1 + ERROR;

        let mut descendant_index = node.descendant_range.start;
        while descendant_index != node.descendant_range.end {
            let descendant = &self.nodes[descendant_index];
            if descendant.depth as usize > depth_threshold {
                debug_assert!(descendant_index < descendant.descendant_range.end);
                // all descendants of `descendant` have a larger depth, so skip them
                descendant_index = descendant.descendant_range.end;
                continue;
            }
            if descendant.character == query[matching_node.checked_len] {
                let in_between = descendant.depth - node.depth - 1;
                //let new_prefix_edit_distance = matching_node.prefix_edit_distance + in_between;

                // this is the edit distance for erasing all characters between the ancestor and descendant
                // which should produce the current query prefix
                let new_edit_distance = matching_node.edit_distance + in_between;

                let next_index = matching_node.checked_len + 1;

                let new_matching_node = MatchingNode::<UUU> {
                    node_id: descendant_index,
                    //query_match_index: next_index,
                    //prefix_edit_distance: new_prefix_edit_distance,
                    checked_len: next_index,
                    edit_distance: new_edit_distance,
                };
                // lines 10-15 from DFA
                if let Some(prior_matching_node) = map.get_mut(descendant) {
                    // this is effectively a way of computing the minimum edit distance between the prefixes
                    if new_edit_distance < prior_matching_node.edit_distance {
                        let removed = priority_queue.remove(prior_matching_node);
                        debug_assert!(removed);

                        let inserted = priority_queue.insert(new_matching_node);
                        debug_assert!(inserted);
                        *prior_matching_node = new_matching_node;
                    }
                } else {
                    // lines 7-9
                    map.insert(descendant, new_matching_node);
                    priority_queue.insert(new_matching_node);
                }
            }
            descendant_index += 1;
        }
    }

    /// Adds all strings prefixed by the node given by `matching_node` to `result`,
    /// or stops when `result` has reached the `requested` size
    ///
    /// Implements Get-MN-STRINGS from DFA
    fn get_matching_node_strings(
        &self,
        matching_node: &MatchingNode<UUU>,
        result: &mut HashMap<&'stored str, MatchingNode<UUU>>,
        requested: usize,
    ) {
        let node = &self.nodes[matching_node.node_id];
        for string_index in node.string_range.clone() {
            let string = self.strings[string_index];
            result.insert(string, matching_node.clone());
            if result.len() == requested {
                break;
            }
        }
    }
}
