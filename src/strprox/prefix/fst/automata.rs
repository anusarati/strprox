use std::collections::{HashMap, HashSet};

use fst::raw::{CompiledAddr, Fst};

// Automaton states and transitions roughly based on formulation by Touzet
// however bit-vectors are not used
// doi:10.1007/978-3-319-30000-9_16

/// Autocompleter that uses the intersection between a non-deterministic
/// Levenshtein automaton and an FST
pub struct A2<D: AsRef<[u8]>> {
    pub index: Fst<D>,
}

/// A state in a non-deterministic Levenshtein automaton
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct State {
    /// Edit distance from the automaton's string
    dist: u8,
    /// Index to the current character
    index: u8,
}

/// Iterator through the strings with the least prefix edit distances from a query
pub struct A2Iter<'a, D: AsRef<[u8]>> {
    index: &'a Fst<D>,
    target: u8,
    query: Vec<char>,
    char_positions: HashMap<char, Vec<usize>>,
    buffer: Vec<(HashSet<State>, String, CompiledAddr)>,
}

impl<D: AsRef<[u8]>> A2Iter<'_, D> {
    /// Returns the next set of states given the `input_char` from the dataset,
    /// whether the minimum edit distance is within the target,
    /// and whether it is an accepting state (based on current target PED)
    fn next_states(
        &self,
        states: &HashSet<State>,
        input_char: char,
    ) -> (HashSet<State>, bool, bool) {
        let mut new_states = HashSet::<State>::with_capacity(2 * states.len());
        let mut target_possible = false;
        let mut target_met = false;
        for state in states {
            if let Some(&query_char) = self.query.get(state.index as usize) {
                // all states below the target should have already been visited
                debug_assert!(state.dist >= self.target);
                if query_char == input_char {
                    // the edit distance has not changed; check the next character later
                    new_states.insert(State {
                        dist: state.dist,
                        index: state.index + 1,
                    });
                    if state.dist == self.target {
                        target_possible = true;
                        // accepting state
                        if state.index == self.query.len() as u8 - 1 { target_met = true; }
                    }
                } else {
                    // possible edits to automaton's strings includes insertion and substitution
                    let insert_state = State {
                        dist: state.dist + 1,
                        // insert `input_char` before `query_char` to match
                        index: state.index,
                    };
                    let sub_state = State {
                        dist: state.dist + 1,
                        index: state.index + 1,
                    };
                    new_states.insert(insert_state);
                    new_states.insert(sub_state);

                    // check for a deletion state (see paper by Touzet)
                    if let Some(positions) = self.char_positions.get(&input_char) {
                        let index = state.index as usize;
                        let next_match_pos = positions.partition_point(|&pos| pos <= index);
                        if next_match_pos < self.query.len() {
                            let deletions = next_match_pos - index;
                            let del_state = State {
                                dist: state.dist + deletions as u8,
                                index: next_match_pos as u8 + 1,
                            };
                            new_states.insert(del_state);
                        }
                    }
                }
            } else {
                // the entire string of the automaton has been checked
                new_states.insert(State {
                    dist: state.dist + 1,
                    index: state.index,
                });
            }
        }
        (new_states, target_possible, target_met)
    }
}
