use std::{
    cmp::min,
    collections::{BTreeSet, HashMap},
    mem,
    ops::{Index, IndexMut},
};

// See the following paper for Levenshtein automata
// Schulz, K., Mihov, S. Fast string correction with Levenshtein automata. IJDAR 5, 67–85 (2002). https://doi.org/10.1007/s10032-002-0082-8

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
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
        other.pos.abs_diff(self.pos) as i16 <= other.dist as i16 - self.dist as i16
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
            pos: idx + 1,
            dist: self.dist + (idx - self.pos),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
/// A state in a Levenshtein DFA
pub struct State {
    positions: Vec<Position>,
}

impl State {
    /// Returns an initial state for the start of the string
    fn new() -> Self {
        State {
            positions: vec![Position::default()],
        }
    }
    /// Removes all positions that are subsumed by another
    fn reduce(&mut self) {
        let mut idx = 0;
        while idx < self.positions.len() {
            self.positions.swap(0, idx);
            let mut jdx = idx + 1;
            while jdx < self.positions.len() {
                if self.positions[0].subsume_eq(self.positions[jdx]) {
                    self.positions.swap_remove(jdx);
                } else {
                    jdx += 1;
                }
            }
            idx += 1;
        }
    }
}

/// An NFA for Levenshtein distance from the `base` string
///
/// `base` may not include '\0'
pub struct LevNFA {
    base: Vec<char>,
    table: InputTable,
}

struct MatchingIndices {
    indices: Vec<u8>,
}

impl MatchingIndices {
    /// Returns the index closest to the position `pos` in the base string
    /// if any
    fn closest_match(&self, pos: u8) -> Option<usize> {
        let point = self.indices.partition_point(|&index| index < pos);
        if point >= self.indices.len() {
            None
        } else {
            Some(self.indices[point] as usize)
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

impl State {
    #[inline]
    /// Inserts the positions after a transition given the current `position`
    /// and the `indices` for characters matching the input
    fn insert_delta(&mut self, position: &Position, indices: Option<&MatchingIndices>) {
        if let Some(indices) = indices {
            let closest = indices.closest_match(position.pos);
            if let Some(idx) = closest {
                if idx as u8 == position.pos {
                    self.positions.push(position.after_match());
                    return;
                } else {
                    self.positions.push(position.after_erase(idx));
                }
            }
        }
        self.positions.push(position.after_replace());
        self.positions.push(position.after_insert());
    }
    /// Returns the new state given the `indices` matching the input character
    fn delta(&self, indices: Option<&MatchingIndices>) -> State {
        let mut next_state = State::default();
        next_state.positions.reserve(3 * self.positions.len());

        for position in &self.positions {
            next_state.insert_delta(position, indices);
        }
        next_state.reduce();
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
    /// Returns length of base string
    pub fn base_len(&self) -> usize {
        self.base.len()
    }
}

/// A state that tracks the minimum known prefix edit distance
#[derive(Debug, Default)]
pub struct PrefixState {
    state: State,
    upper_bound: u8,
}

impl PrefixState {
    /// Returns a state for the beginning of the string
    pub fn new() -> Self {
        PrefixState {
            state: State::new(),
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
    /// Returns the prefix distance assuming an input string has been entirely consumed
    pub fn final_prefix_dist(&self, query_len: u8) -> u8 {
        // this checks to see what prefix distance can be achieved by filling in missing
        // characters, in case the data string is shorter than the query
        min(
            self.upper_bound,
            self.state
                .positions
                .iter()
                .map(|position| query_len.abs_diff(position.pos) + position.dist)
                .min()
                .unwrap_or(u8::MAX),
        )
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
    /// Returns length of base string
    pub fn base_len(&self) -> usize {
        self.base.base_len()
    }
}
