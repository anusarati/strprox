use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    ops::Range,
};

use fst::{
    raw::{Fst, Node, Transition},
    Set, Streamer,
};

pub mod automata;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{levenshtein, MeasuredPrefix};

use super::{Autocompleter, PrefixRanking, PrefixRankings, FromStrings};


/// Iterator over the next characters and associated states from a node in a FST
struct NodeCharIterator<'f, 'n, D: AsRef<[u8]>> {
    /// FST used to retrieve the nodes associated with the next characters
    fst: &'f Fst<D>,
    /// Nodes before the transitions
    nodes: [Option<Node<'n>>; 4],
    /// Indices for possible transitions needed to fill a character
    transition_indices: [u8; 4],
    /// Bytes used to decode UTF-8
    bytes: [u8; 4],
    /// Last index of byte needed to decode a character in UTF-8
    last: u8,
}

impl<'f: 'n, 'n: 'f, D: AsRef<[u8]>> NodeCharIterator<'f, 'n, D> {
    /// Returns the char at the iterator
    fn get_char(&self) -> char {
        std::str::from_utf8(&self.bytes[0..=self.last as usize])
            .unwrap()
            .chars()
            .next()
            .unwrap()
    }
    /// Determines the number of bytes needed to decode a UTF-8 character
    fn classify_first_byte(&mut self, byte: u8) {
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
    /// Initializes subsequent nodes for decoding UTF-8 based on the first transition
    fn initialize_subsequent_nodes(&mut self, mut transition: Transition) {
        for i in 1..=self.last as usize {
            let node = self.fst.node(transition.addr);
            self.nodes[i] = Some(node);
            //self.transition_indices[i] = 0;
            debug_assert_eq!(self.transition_indices[i], 0);
            // get the first transition of the next node
            transition = self.fst.node(transition.addr).transition(0);
        }
    }
    /// Prepare to decode UTF-8 based on the first transition
    fn set_first_transition(&mut self, transition: Transition) {
        self.classify_first_byte(transition.inp);
        self.initialize_subsequent_nodes(transition);
    }
    /// Returns an iterator over all characters and associated nodes after the `node` in `fst`
    fn new(fst: &'f Fst<D>, node: &'f Node<'n>) -> Self {
        Self {
            fst,
            nodes: [Some(node.clone()), None, None, None],
            transition_indices: [0; 4],
            bytes: [0; 4],
            last: 0,
        }
    }
}

impl<'f: 'n, 'n: 'f, D: AsRef<[u8]>> Iterator for NodeCharIterator<'f, 'n, D> {
    type Item = (Node<'n>, char);
    /// Returns Option with the next Node and character if possible
    fn next(&mut self) -> Option<Self::Item> {
        let mut index = self.last as usize;
        loop {
            let transition_index = self.transition_indices[index] as usize;
            let node = self.nodes[index].unwrap();
            // if it's not the last transition from the node
            if transition_index < node.len() {
                self.transition_indices[index] += 1;
                let transition = node.transition(transition_index);
                self.bytes[index] = transition.inp;
                if index == 0 {
                    self.set_first_transition(transition);
                }
                if index == self.last as usize {
                    return Some((self.fst.node(transition.addr), self.get_char()));
                } else {
                    index += 1;
                }
            } else {
                if index == 0 {
                    return None;
                } else {
                    self.transition_indices[index] = 0;
                    index -= 1;
                }
            }
        }
    }
}