use std::{
    collections::{BTreeSet, HashMap},
    ops::{Index, IndexMut},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
/// A state in a Levenshtein NFA
struct Position {
    pos: u8,
    dist: u8,
}

impl Position {
    fn subsumes(&self, other: Position) -> bool {
        other.dist > self.dist && other.pos.abs_diff(self.pos) <= other.dist - self.dist
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
/// A state in a Levenshtein DFA
struct State {
    positions: BTreeSet<Position>,
}

#[derive(Default)]
struct States {
    /// Numeric labels for each state
    numbers: HashMap<*const State, usize>,
    states: Vec<State>,
}

impl States {
    /// Returns the number of `state`
    fn insert(&mut self, state: State) -> usize {
        self.states.push(state);
        self.numbers
            .insert(self.states.last().unwrap(), self.states.len());
        self.states.len() - 1
    }
}

/// A DFA for Levenshtein distance from the `base` string
///
/// `base` may not include '\0'
struct LevDFA {
    base: Vec<char>,

    /// Tracked states labeled with numbers
    current_states: States,
    next_states: States,

    transitions: HashMap<(usize, Input), usize>,
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

/// Maps a character to an input bit-vector
struct InputTable {
    table: HashMap<char, Input>,
}

impl InputTable {
    /// Returns input table that maps all characters in a Vec<char> to a characteristic vector
    fn new(base: &Vec<char>) -> Self {
        let mut table = HashMap::<char, Input>::new();
        for (index, &character) in base.iter().enumerate() {
            if let Some(input) = table.get_mut(&character) {
                input.matches[index] = true;
            } else {
                let mut matches = [false; 256];
                matches[index] = true;
                table.insert(character, Input { matches });
            }
        }
        Self { table }
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

impl LevDFA {
    /// Returns an empty LevDFA for the `base` string
    pub(super) fn new(base: String) -> Self {
        let base = base.chars().collect();
        let table = InputTable::new(&base);
        Self {
            base,
            current_states: Default::default(),
            next_states: Default::default(),
            transitions: Default::default(),
            table,
        }
    }
    /// Adds the transition from `state` and the `current` character in the other string
    /// to the automaton; does nothing if it already exists
    pub(super) fn add_transition(&mut self, state: &State, input: char) {
        let characteristic = self.table[input];
        let number = *self.current_states.numbers.get(state).unwrap();
        if !self.transitions.contains_key(&(number, characteristic)) {
            let next = self.delta(number, characteristic);
            let next_num = self.next_states.insert(next);
            self.transitions.insert((number, characteristic), next_num);
        }
    }
    /// Returns the new state given a characteristic input vector
    fn delta(&self, state_num: usize, characteristic: Input) -> State {}
}
