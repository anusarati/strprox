use std::{
    cmp::min,
    collections::{BTreeSet, HashMap},
    ops::{Index, IndexMut},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
/// A state in a Levenshtein NFA
struct Position {
    /// The position in the base string (length of prefix)
    pos: u8,
    /// The edit distance from the base prefix
    dist: u8,
}

impl Position {
    /// Returns whether this position makes `other` redundant
    /// (subsume or equal)
    fn subsume_eq(&self, other: Position) -> bool {
        other.dist >= self.dist && other.pos.abs_diff(self.pos) <= other.dist - self.dist
    }
    /// Returns the position after matching the current character
    fn after_match(&self) -> Position {
        Position {
            pos: self.pos + 1,
            dist: self.dist,
        }
    }
    /// Returns the position after replacing the current character
    fn after_replace(&self) -> Position {
        Position {
            pos: self.pos + 1,
            dist: self.dist + 1,
        }
    }
    /// Returns the position after inserting at the current position
    fn after_insert(&self) -> Position {
        Position {
            pos: self.pos,
            dist: self.dist + 1,
        }
    }
    /// Returns the position after erasing until `matching_idx` is reached in the string
    fn after_erase(&self, matching_idx: usize) -> Position {
        let idx = matching_idx as u8;
        Position {
            pos: idx,
            dist: self.dist + (idx - self.pos),
        }
    }
}

impl PartialOrd for Position {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Compare by edit distance and then by position in base string
/// to find subsumed states quickly
impl Ord for Position {
    // idea based on Dylon Edwards' liblevenshtein
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .cmp(&other.dist)
            .then_with(|| self.pos.cmp(&other.pos))
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
/// A state in a Levenshtein DFA
struct State {
    positions: Vec<Position>,
}

impl State {
    /// Inserts a position into the deterministic state, removing all redundant positions
    fn insert(&mut self, position: Position) {
        let mut idx = 0;
        while idx < self.positions.len() {
            if position.subsume_eq(self.positions[idx]) {
                self.positions.swap_remove(idx);
            } else {
                idx += 1;
            }
        }
        self.positions.push(position)
    }
}

/// An NFA for Levenshtein distance from the `base` string
///
/// `base` may not include '\0'
struct LevNFA {
    base: Vec<char>,
    table: InputTable,
}

/// A bit-vector corresponding to which characters in a string match
/// used as input to the automaton (aka characteristic vector)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Input {
    matches: [bool; 256],
}

impl Default for Input {
    fn default() -> Self {
        Self {
            matches: [false; 256],
        }
    }
}

impl Index<usize> for Input {
    type Output = bool;
    fn index(&self, index: usize) -> &Self::Output {
        &self.matches[index]
    }
}

struct MatchingIndices {
    indices: Vec<u8>,
}

impl MatchingIndices {
    /// Returns the index closest to the position `pos` in the base string
    /// if any
    fn closest_match(&self, pos: u8) -> Option<usize> {
        let idx = self.indices.partition_point(|&index| index < pos);
        if idx >= self.indices.len() {
            None
        } else {
            Some(idx)
        }
    }
    fn new(index: usize) -> Self {
        Self {
            indices: vec![index as u8],
        }
    }
    fn push(&mut self, index: usize) {
        self.indices.push(index as u8)
    }
}

/// Maps a character to its matching indices in a base string
struct InputTable {
    map: HashMap<char, MatchingIndices>,
}

impl InputTable {
    /// Returns table that maps all characters in a string to a characteristic vector
    fn new(base: &Vec<char>) -> Self {
        let mut map = HashMap::<char, MatchingIndices>::new();
        for (index, &character) in base.iter().enumerate() {
            if let Some(indices) = map.get_mut(&character) {
                indices.push(index);
            } else {
                map.insert(character, MatchingIndices::new(index));
            }
        }
        Self { map }
    }
    /// Returns indices into base string where the character equals `input`,
    /// if any
    fn matching_indices(&self, input: char) -> Option<&MatchingIndices> {
        self.map.get(&input)
    }
}

impl Position {
    /// Returns the next possible positions from this position
    /// given the index of the closest matching character to the input
    fn delta(&self, indices: Option<&MatchingIndices>) -> Vec<Position> {
        if let Some(indices) = indices {
            let closest = indices.closest_match(self.pos);
            match closest {
                Some(idx) => {
                    if idx as u8 == self.pos {
                        vec![self.after_match()]
                    } else {
                        vec![
                            self.after_replace(),
                            self.after_insert(),
                            self.after_erase(idx),
                        ]
                    }
                }
                None => {
                    vec![self.after_replace(), self.after_insert()]
                }
            }
        } else {
            vec![self.after_replace(), self.after_insert()]
        }
    }
}

impl State {
    /// Returns the new state given the indices matching the input character
    fn delta(&self, indices: Option<&MatchingIndices>) -> State {
        let mut next_state = State::default();
        for position in &self.positions {
            for next_position in position.delta(indices) {
                next_state.insert(next_position)
            }
        }
        next_state
    }
    /// Returns the minimum edit distance from the `base` string based on the input received
    pub fn min_dist(&self) -> u8 {
        self.positions
            .iter()
            .map(|position| position.dist)
            .min()
            .unwrap_or(u8::MAX)
    }
}

impl LevNFA {
    /// Returns an empty LevNFA for the `base` string
    pub fn new(base: &str) -> Self {
        let base = base.chars().collect();
        let table = InputTable::new(&base);
        Self { base, table }
    }
    /// Returns the next state given the input
    pub fn delta(&self, current: &State, input: char) -> State {
        current.delta(self.table.matching_indices(input))
    }
}

/// A state that tracks the minimum known prefix edit distance
pub struct PrefixState {
    state: State,
    upper_bound: u8,
}

impl PrefixState {
    pub fn new() -> Self {
        PrefixState {
            state: State::default(),
            upper_bound: u8::MAX,
        }
    }
    /// Returns upper bound on the prefix distance between the base and input string
    pub fn prefix_dist_bound(&self) -> u8 {
        self.upper_bound
    }
    /// Returns the minimum prefix edit distance from the base string based on the input received
    pub fn min_dist(&self) -> u8 {
        min(self.state.min_dist(), self.upper_bound)
    }
}

/// Levenshtein NFA that tracks the prefix edit distance
pub struct PrefixLevNFA {
    base: LevNFA,
}

impl PrefixLevNFA {
    /// Returns NFA with states with edit distance information from the `base_string`
    pub fn new(base_string: &str) -> Self {
        Self {
            base: LevNFA::new(base_string),
        }
    }
    /// Returns the next state given the current state and an input character
    pub fn delta(&self, current: &PrefixState, input: char) -> PrefixState {
        let state = self.base.delta(&current.state, input);

        let upper_bound = min(
            current.prefix_dist_bound(),

            // unlike state.min_dist(), this only changes the upper bound
            // if an edit distance (maybe unoptimal) is known for the entire base string
            state
                .positions
                .iter()
                .filter(|position| position.pos == self.base.base.len() as u8)
                .map(|position| position.dist)
                .min()
                .unwrap_or(u8::MAX),
        );

        PrefixState { state, upper_bound }
    }
}
