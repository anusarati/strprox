use std::{
    io::Write,
    time::{Duration, Instant},
};

use crate::strprox::Autocompleter;

#[test]
/// Example input from the DFA paper
fn dfa_paper_example() {
    let source = vec![
        "geem", "genea", "genep", "genez", "genome", "genet", "gele", "ner",
    ];
    let autocompleter = Autocompleter::<u8>::new(&source);
    let result = autocompleter.autocomplete("gne", 3);
    for measure in &result {
        println!("{:#?}", measure);
    }
    let _ = std::io::stdout().flush();
    assert!(result.iter().any(|measure| { measure.0 == "geem" }));
    assert!(result.iter().any(|measure| { measure.0 == "genea" }));
    // the paper has "genep" instead of "gele"
    // I believe this may be because this implementation sorts the 0s
    // all the 0s do in fact have the same prefix edit distance from "gne" (1)
    assert!(result.iter().any(|measure| { measure.0 == "gele" }));
}

#[test]
/// Toy test where the prefix edit distances are different
fn two_categories() {
    let source = vec![
        "banana",
        "labana",
        "alabama",
        "orange",
        "blorange",
        "orangutan",
        "range",
    ];
    let autocompleter = Autocompleter::<u8>::new(&source);
    let result = autocompleter.autocomplete("ababa", 3);
    println!("ababa\n");
    for measure in &result {
        println!("{:#?}", measure);
    }
    let _ = std::io::stdout().flush();
    assert!(result.iter().any(|measure| { measure.0 == "banana" })); // && measure.prefix_distance == 2 }));
    assert!(result.iter().any(|measure| { measure.0 == "labana" })); // && measure.prefix_distance == 2 }));
    assert!(result.iter().any(|measure| { measure.0 == "alabama" })); //&& measure.prefix_distance == 1 }));

    let result = autocompleter.autocomplete("oange", 4);
    println!("oange\n");
    for measure in &result {
        println!("{:#?}", measure);
    }
    assert!(result.iter().any(|measure| { measure.0 == "orange" }));
    assert!(result.iter().any(|measure| { measure.0 == "blorange" }));
    assert!(result.iter().any(|measure| { measure.0 == "orangutan" }));
    assert!(result.iter().any(|measure| { measure.0 == "range" }));
}

/// Test that autocomplete works for a prefix that only requires an insertion at the beginning
#[test]
fn insertion_ped() {
    let query = "foob";
    // PEDs: [1, 2, 2]
    let source = vec!["oobf", "fbor", "bobf"];
    let autocompleter = Autocompleter::<u8>::new(&source);
    let result = autocompleter.autocomplete(query, 1);
    for measure in &result {
        println!("{:#?}", measure);
    }
    assert_eq!(result[0].0, "oobf");
}

const WORDS: &str = include_str!("words.txt");

#[test]
/// Tests for correction of a misspelling against a large database
fn large_database_misspelling() {
    let source: Vec<&str> = WORDS.lines().collect();
    let time = Instant::now();
    let autocompleter = Autocompleter::<u8>::new(&source);
    println!("Construction of trie took: {:#?}", time.elapsed());
    let requested = 10;
    let result = autocompleter.autocomplete("abandonned", requested);
    assert_eq!(result.len(), requested);

    for measure in &result {
        println!("{:#?}", measure);
    }
    assert!(result.iter().any(|measure| { measure.0 == "abandoned" }));
}

#[test]
/// Tests for error-tolerant autocompletion against a large database
fn large_database_autocomplete() {
    let source: Vec<&str> = WORDS.lines().collect();
    let autocompleter = Autocompleter::<u8>::new(&source);
    let requested = 10;
    let result = autocompleter.autocomplete("oberr", requested);
    assert_eq!(result.len(), requested);

    for measure in &result {
        println!("{:#?}", measure);
    }
    assert!(result.iter().any(|measure| { measure.0 == "aberration" }));
}
