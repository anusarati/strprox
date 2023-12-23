use std::{
    io::Write,
    time::{Duration, Instant},
};

use crate::strprox::Autocompleter;

#[test]
/// Example input from the paper on META
fn meta_paper_example() {
    let source = vec!["soho", "solid", "solo", "solve", "soon", "throw"];
    let autocompleter = Autocompleter::<u8>::new(&source);
    let result = autocompleter.autocomplete("ssol", 3);
    for measure in &result {
        println!("{:#?}", measure);
    }
    let _ = std::io::stdout().flush();
    assert!(result.iter().any(|measure| { measure.string == "solid" }));
    assert!(result.iter().any(|measure| { measure.string == "solo" }));
    assert!(result.iter().any(|measure| { measure.string == "solve" }));
}

#[test]
/// Tests that autocomplete can return exact associated categories
fn two_categories() {
    let source = vec![
        "success",
        "successor",
        "successive",
        "decrement",
        "decrease",
        "decreasing",
    ];
    let autocompleter = Autocompleter::<u8>::new(&source);
    let query = "zucc";
    let result = autocompleter.autocomplete(query, 3);
    println!("{}\n", query);
    for measure in &result {
        println!("{:#?}", measure);
    }
    let _ = std::io::stdout().flush();
    assert!(result.iter().any(|measure| { measure.string == "success" })); // && measure.prefix_distance == 2 }));
    assert!(result.iter().any(|measure| { measure.string == "successor" })); // && measure.prefix_distance == 2 }));
    assert!(result.iter().any(|measure| { measure.string == "successive" })); //&& measure.prefix_distance == 1 }));

    let query = "deck";
    let result = autocompleter.autocomplete("deck", 3);
    println!("{}\n", query);
    for measure in &result {
        println!("{:#?}", measure);
    }
    assert!(result.iter().any(|measure| { measure.string == "decrement" }));
    assert!(result
        .iter()
        .any(|measure| { measure.string == "decrease" }));
    assert!(result
        .iter()
        .any(|measure| { measure.string == "decreasing" }));
}

/// Tests that autocomplete works for a prefix that only requires an insertion at the beginning
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
    assert_eq!(result[0].string, "oobf");
}

const WORDS: &str = include_str!("words.txt");

#[test]
/// Tests for correction of a misspelling against a large database
fn large_database_misspelling() {
    let source: Vec<&str> = WORDS.lines().collect();
    let time = Instant::now();
    let autocompleter = Autocompleter::<u8>::new(&source);
    println!("Indexing took: {:#?}", time.elapsed());
    let requested = 10;
    let result = autocompleter.autocomplete("abandonned", requested);
    assert_eq!(result.len(), requested);

    for measure in &result {
        println!("{:#?}", measure);
    }
    assert!(result
        .iter()
        .any(|measure| { measure.string == "abandoned" }));
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
    assert!(result
        .iter()
        .any(|measure| { measure.string == "overrank" }));
}
