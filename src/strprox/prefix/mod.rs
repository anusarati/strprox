use priority_queue::PriorityQueue;
use std::{
    cmp::{max, min, Reverse},
    collections::{BTreeSet, HashMap},
    ops::Range,
    string,
};

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

impl Node<UUU> {
    /// Returns the index into the trie where the node is
    #[inline]
    fn id(&self) -> usize {
        self.descendant_range.start - 1
    }
    #[inline]
    /// Returns a value equivalent to the id for sorting
    fn sort_id(&self) -> usize {
        self.descendant_range.start
    }
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
}

/// Inverted index from META
struct InvertedIndex<'stored, UUU> {
    /// depth |-> (character |-> nodes in trie)
    index: Vec<HashMap<char, Vec<&'stored Node<UUU>>>>,
}

impl<'stored> InvertedIndex<'stored, UUU> {
    /// Constructs an inverted index from depth to character to nodes using a trie
    fn new(trie: &Trie<'stored, UUU>) -> Self {
        let mut max_depth = 0;
        for node in &trie.nodes {
            max_depth = max(max_depth, node.depth as usize);
        }

        let mut index = Vec::<HashMap<char, Vec<&'stored Node<UUU>>>>::with_capacity(max_depth + 1);
        index.resize(max_depth + 1, Default::default());

        // put all nodes into the index at a certain depth and character
        for node in &trie.nodes {
            let depth = node.depth as usize;
            let char_map = &mut index[depth];
            if let Some(nodes) = char_map.get_mut(&node.character) {
                nodes.push(&node);
            } else {
                char_map.insert(node.character, vec![&node]);
            }
        }
        // sort the nodes by id for binary search (cache locality with Vec)
        for char_map in &mut index {
            for (_, nodes) in char_map {
                nodes.sort_by(|first, second| first.sort_id().cmp(&second.sort_id()));
            }
        }
        Self { index }
    }
}

/// Structure that allows for autocompletion based on a string dataset
struct Autocompleter<'stored, UUU> {
    trie: Trie<'stored, UUU>,
    inverted_index: InvertedIndex<'stored, UUU>,
}

impl<'stored> Autocompleter<'stored, UUU> {
    /// Constructs an Autocompleter given the string dataset `source` (does not copy strings)
    fn new(source: &[&'stored str]) -> Self {
        let trie = Trie::<'stored, UUU>::new(source);
        let inverted_index = InvertedIndex::<'stored, UUU>::new(&trie);
        Self {
            trie,
            inverted_index,
        }
    }
    /// Returns the top `requested` number of strings with the best prefix distance from the query,
    /// or all strings sorted by prefix edit distance if `requested` is larger than the number stored
    pub fn autocomplete(&self, query: &str, requested: usize) -> Vec<MeasuredPrefix> {
        if requested == 0 {
            return Default::default();
        }
        "foo"
    }
}
