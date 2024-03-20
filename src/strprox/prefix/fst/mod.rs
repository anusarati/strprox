use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    mem,
    ops::Range,
};

use fst::{
    map::StreamBuilder,
    raw::{Fst, Node, Transition},
    Set, Streamer,
};

pub mod automata;
use automata::{PrefixLevNFA, PrefixState};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{Autocompleter, MeasuredPrefix};

use super::FromStrings;

/// A position in a FST, used to retrieve the node associated with the next character
#[derive(Clone, Copy)]
struct FstPosition<'n> {
    root: Node<'n>,
    /// Indices for possible transitions needed to fill a character
    transition_indices: [u8; 4],
    /// Bytes used to decode UTF-8
    bytes: [u8; 4],
    /// Last index of byte needed to decode a character in UTF-8
    last: u8,
}

impl<'n> FstPosition<'n> {
    /// Returns the char at the iterator
    fn get_char(&self) -> char {
        std::str::from_utf8(&self.bytes[0..=self.last as usize])
            .unwrap()
            .chars()
            .next()
            .unwrap()
    }
    /// Determines the number of bytes needed to decode a UTF-8 character
    fn classify_first_byte(&mut self) {
        let byte = self.bytes[0];
        // https://en.wikipedia.org/wiki/UTF-8#Encoding
        self.last = if byte < 0b11000000 {
            0
        } else if byte < 0b11100000 {
            1
        } else if byte < 0b11110000 {
            2
        } else {
            3
        }
    }
    /// Gets the node after `index` transitions
    fn get_node<D: AsRef<[u8]>>(&self, fst: &'n Fst<D>, index: usize) -> Node<'n> {
        let mut node = self.root;
        for i in 0..index {
            node = fst.node(node.transition(self.transition_indices[i] as usize).addr);
        }
        node
    }
    /// Returns a position at the FST `node`
    fn new(root: Node<'n>) -> Self {
        Self {
            root,
            transition_indices: [0; 4],
            bytes: [0; 4],
            last: 0,
        }
    }
}

/// Iterator over the next characters and associated states from a node in a FST
struct NodeCharIterator<'f, 'p, D: AsRef<[u8]>> {
    fst: &'f Fst<D>,
    position: &'p mut FstPosition<'f>,
}

impl<'f, 'p, D: AsRef<[u8]>> Iterator for NodeCharIterator<'f, 'p, D> {
    type Item = (Node<'f>, char);
    /// Returns Option with the next Node and character if possible
    fn next(&mut self) -> Option<Self::Item> {
        let position = &mut self.position;
        let mut index = position.last as usize;
        let mut node = position.get_node(self.fst, index);
        loop {
            let transition_index = position.transition_indices[index] as usize;
            let node = position.get_node(self.fst, index);
            // if it's not the last transition from the node
            if transition_index < node.len() {
                position.transition_indices[index] += 1;
                let transition = node.transition(transition_index);
                position.bytes[index] = transition.inp;
                if index == 0 {
                    position.classify_first_byte();
                }
                if index == position.last as usize {
                    return Some((self.fst.node(transition.addr), position.get_char()));
                } else {
                    index += 1;
                }
            } else {
                if index == 0 {
                    return None;
                } else {
                    position.transition_indices[index] = 0;
                    index -= 1;
                }
            }
        }
    }
}

/// An iterator over the strings in an FST with the closest prefixes to a query
pub struct AutocompleteIter<'index, D: AsRef<[u8]>> {
    index: &'index Fst<D>,
    /// Used to track prefix edit distance states
    automaton: PrefixLevNFA,
    /// The strings whose edit distances are at least `threshold`
    candidates: Vec<(PrefixState, String, FstPosition<'index>)>,
    /// The strings whose edit distances exceed `threshold`
    buffer: Vec<(PrefixState, String, FstPosition<'index>)>,
    /// Tracks the next strings to be returned with their prefixes
    /// and suffix positions
    mark: Vec<(String, FstPosition<'index>)>,
    /// The minimum prefix edit distance for unreturned strings
    target: u8,
    /// The maximum tolerable prefix edit distance
    threshold: u8,
}

impl<'index, D: AsRef<[u8]>> AutocompleteIter<'index, D> {
    pub fn new(index: &'index Fst<D>, query: &str) -> Self {
        Self {
            index,
            automaton: PrefixLevNFA::new(query),
            candidates: vec![(
                PrefixState::new(),
                String::new(),
                FstPosition::<'index>::new(index.root()),
            )],
            buffer: vec![],
            mark: vec![],
            target: 0,
            threshold: u8::MAX,
        }
    }
    /// Sets the maximum tolerable prefix edit distance
    pub fn set_threshold(&mut self, threshold: u8) {
        self.threshold = threshold;
    }
    /// Returns the next string that was marked as meeting the threshold
    fn pop_mark(&mut self) -> Option<MeasuredPrefix> {
        while let Some((ref prefix, position)) = self.mark.last_mut() {
            let mut iter = NodeCharIterator {
                fst: self.index,
                position,
            };
            // DFS, stop at a terminal node
            if let Some((node, input)) = iter.next() {
                let mut next_pre = prefix.clone();
                next_pre.push(input);

                self.mark.push((next_pre, FstPosition::new(node)));
            } else if position.root.is_final() {
                let res = Some(MeasuredPrefix {
                    string: prefix.to_owned(),
                    prefix_distance: self.target as usize,
                });
                self.mark.pop();
                return res;
            } else {
                self.mark.pop();
            }
        }
        None
    }
}

impl<'index, D: AsRef<[u8]>> Iterator for AutocompleteIter<'index, D> {
    type Item = MeasuredPrefix;

    /// Returns the next closest string, if any
    fn next(&mut self) -> Option<Self::Item> {
        // an edge case is when the query is empty, which means all strings have 0
        // prefix edit distance
        if self.automaton.base_len() == 0 && !self.candidates.is_empty() {
            // this must be the initial state because this branch can only execute once per iterator
            // at the beginning
            self.mark.push(
                self.candidates
                    .pop()
                    .map(|(_, string, position)| (string, position))
                    .unwrap(),
            );
        }

        if let Some(result) = self.pop_mark() {
            return Some(result);
        }

        let mut new_candidate = None;
        let mut ret = None;
        loop {
            if ret.is_some() {
                return ret;
            }

            if let Some(candidate) = mem::take(&mut new_candidate) {
                self.candidates.push(candidate);
            } else if self.candidates.is_empty() {
                if self.buffer.is_empty() {
                    return None;
                }
                // try checking for strings with a higher prefix edit distance
                mem::swap(&mut self.candidates,&mut self.buffer);
                self.buffer.clear();

                self.target += 1;

                // can't find anything better
                if self.target > self.threshold {
                    return None;
                }

                self.buffer.reserve(self.candidates.len());
            }

            while let Some((state, string, position)) = self.candidates.last_mut() {
                // evaluate all extended strings
                let iter = NodeCharIterator {
                    fst: self.index,
                    position,
                };

                for (node, input) in iter {
                    let next_state = self.automaton.delta(state, input);
                    let mut next_string = string.clone();
                    next_string.push(input);
                    let next_position = FstPosition::new(node);

                    // the PED must be equal to the target because we exhausted lower PEDs
                    if next_state.prefix_dist_bound() <= self.target {
                        self.mark.push((next_string, next_position));
                        return self.pop_mark();
                    }

                    let lower_bound = next_state.min_dist();
                    //println!("{:#} {:#} {:#?}", lower_bound, next_string, next_state);
                    let next_tuple = (next_state, next_string, next_position);

                    // can't possibly meet the target PED
                    if lower_bound > self.target {
                        self.buffer.push(next_tuple);
                    } else {
                        new_candidate = Some(next_tuple);
                        break;
                    }
                }

                if new_candidate.is_some() {
                    break;
                }

                if position.root.is_final() {
                    // the data string might be shorter than the query, so we need to check what its prefix
                    // edit distance would be if we filled in the rest
                    if state.final_prefix_dist(self.automaton.base_len() as u8) <= self.target {
                        ret = Some(MeasuredPrefix {
                            string: string.to_owned(),
                            prefix_distance: self.target as usize,
                        });
                    } else {
                        // delay emitting this string
                        self.buffer
                            .push((mem::take(state), mem::take(string), position.clone()));
                    }
                }

                // finished checking this node
                self.candidates.pop();
            }
        }
    }
}

/// Top-k autocomplete using NFA for tests
pub struct NFAAutocompleter<D: AsRef<[u8]>> {
    index: Fst<D>,
}

impl<D: AsRef<[u8]>> NFAAutocompleter<D> {
    pub fn new(index: Fst<D>) -> Self {
        Self { index }
    }
}

impl<D: AsRef<[u8]>> Autocompleter for NFAAutocompleter<D> {
    #[allow(refining_impl_trait)]
    fn top_iter(&self, query: &str) -> AutocompleteIter<'_, D> {
        AutocompleteIter::new(&self.index, query)
    }
    fn threshold_topk(
        &self,
        query: &str,
        requested: usize,
        max_threshold: usize,
    ) -> Vec<MeasuredPrefix> {
        let mut iter = AutocompleteIter::new(&self.index, query);
        iter.set_threshold(max_threshold as u8);
        let mut result: Vec<_> = iter.take(requested).collect();
        result.sort();
        result
    }
}

impl FromStrings for NFAAutocompleter<Vec<u8>> {
    fn from_strings(strings: &[&str]) -> Self {
        let mut source = strings.to_owned();
        source.sort();
        Self {
            index: Set::from_iter(source).unwrap().into_fst(),
        }
    }
}
