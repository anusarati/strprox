use std::{
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
    /// Returns the position after inserting at the current position
    fn after_insert(&self) -> Position {
        Position {
            pos: self.pos,
            dist: self.dist + 1,
        }
    }
    /// Returns the position after erasing at the current position
    fn after_erase(&self, characteristic: &Input) -> Position {
        let next_char_different = !characteristic[self.pos as usize + 1] as u8;
        Position {
            pos: self.pos + 1,
            dist: self.dist + 1 + next_char_different,
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

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
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

/// Maps a character to an input bit-vector
struct InputTable {
    characteristic_table: HashMap<char, Input>,
    matching_indices: HashMap<char, Vec<u8>>,
}

impl InputTable {
    /// Returns table that maps all characters in a string to a characteristic vector
    fn new(base: &Vec<char>) -> Self {
        let mut characteristic_table = HashMap::<char, Input>::new();
        let mut matching_indices = HashMap::<char, Vec<u8>>::new();
        for (index, &character) in base.iter().enumerate() {
            if let Some(indices) = matching_indices.get_mut(&character) {
                input.matches[index] = true;
                matching_indices[&character].push(index as u8);
            } else {
                let mut matches = [false; 256];
                matches[index] = true;
                characteristic_table.insert(character, Input { matches });
                matching_indices.insert(character, vec![index as u8]);
            }
        }
        Self {
            characteristic_table,
            matching_indices,
        }
    }
    /// Returns index of closest character in base string that equals `input`,
    /// if any
    fn closest_match(&self, input: char, index: usize) -> Option<usize> {
        if let Some(indices) = self.matching_indices.get(&input) {
            let idx = indices.partition_point(|&matching| matching < index as u8);
            if idx >= indices.len() {
                None
            } else {
                Some(idx)
            }
        } else {
            None
        }
    }
}

const NULL_INPUT: Input = Input {
    matches: [false; 256],
};

impl Index<char> for InputTable {
    type Output = Input;
    /// Returns the characteristic vector for the `current` character
    ///
    /// If `current` was not in the original string, returns a null vector
    fn index(&self, current: char) -> &Input {
        self.table.get(&current).unwrap_or(&NULL_INPUT)
    }
}
impl Position {
    /// Returns the next possible positions given a position and an input
    fn delta(&self, characteristic: &Input) -> Vec<Position> {
        if characteristic[self.pos as usize] {
            vec![self.after_match()]
        } else {
            vec![self.after_insert(), self.after_erase(characteristic)]
        }
    }
}
impl State {
    /// Returns the new state given a characteristic input vector
    fn delta(&self, current: &State, characteristic: &Input) -> State {
        todo!()
    }
}
impl LevNFA {
    /// Returns an empty LevDFA for the `base` string
    pub(super) fn new(base: String) -> Self {
        let base = base.chars().collect();
        let table = InputTable::new(&base);
        Self { base, table }
    }
}
