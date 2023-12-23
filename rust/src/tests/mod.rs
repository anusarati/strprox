use std::{
    cmp::min,
    io::Write,
    time::{Duration, Instant},
};

use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

use crate::{
    levenshtein::{prefix_edit_distance, random_edits, random_string},
    strprox::Autocompleter,
};

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
    assert!(result
        .iter()
        .any(|measure| { measure.string == "successor" })); // && measure.prefix_distance == 2 }));
    assert!(result
        .iter()
        .any(|measure| { measure.string == "successive" })); //&& measure.prefix_distance == 1 }));

    let query = "deck";
    let result = autocompleter.autocomplete("deck", 3);
    println!("{}\n", query);
    for measure in &result {
        println!("{:#?}", measure);
    }
    assert!(result
        .iter()
        .any(|measure| { measure.string == "decrement" }));
    assert!(result
        .iter()
        .any(|measure| { measure.string == "decrease" }));
    assert!(result
        .iter()
        .any(|measure| { measure.string == "decreasing" }));
}

#[test]
/// The example in the README
fn example() {
    let source = vec![
        "success",
        "successive",
        "successor",
        "decrease",
        "decreasing",
        "decrement",
    ];
    let autocompleter = Autocompleter::new(&source);
    let query = "luck";
    let result = autocompleter.autocomplete(query, 3);
    for measured_prefix in &result {
        println!("{}", measured_prefix);
    }
    let result_strings: Vec<&str> = result
        .iter()
        .map(|measured_prefix| measured_prefix.string.as_str())
        .collect();
    assert_eq!(result_strings, vec!["success", "successive", "successor"]);
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

// Words from https://github.com/dwyl/english-words/blob/master/words.txt
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

#[test]
/// Tests that the result from an empty query still has strings
fn empty_query_test() {
    let source: Vec<&str> = WORDS.lines().collect();
    let autocompleter = Autocompleter::<u8>::new(&source);
    let result = autocompleter.autocomplete("", 1);
    assert_ne!(result.len(), 0);
}

#[test]
/// Tests that prefix edit distances are within the number of edits made to strings from a database
/// using 1000 random data points
fn large_database_bounded_peds() {
    let source: Vec<&str> = WORDS.lines().collect();
    let autocompleter = Autocompleter::<u8>::new(&source);
    let mut rng = rand::thread_rng();
    let mut total_duration = Duration::new(0, 0);
    for _i in 0..1e3 as usize {
        let string = random_string(&source[..]);
        let edits_distribution = Uniform::new(0, 5);
        let edits = edits_distribution.sample(&mut rng);
        let edited_string = random_edits(string, edits);

        dbg!(&edited_string);

        let time = Instant::now();
        let result = &autocompleter.autocomplete(edited_string.as_str(), 1)[0];
        total_duration += time.elapsed();

        dbg!(result);

        // Depending on what edits were made, the result may not necessarily be equal to `string` (e.g. 5 edits to a string with a length of 5)
        // so we do not check that

        assert_eq!(
            prefix_edit_distance(edited_string.as_str(), result.string.as_str()),
            result.prefix_distance,
            "Prefix edit distance incorrect"
        );
        assert!(
            result.prefix_distance <= edits,
            "Resulting prefix edit distance not bounded by edits made"
        );
    }
    println!(
        "Average time per query: {} ms",
        total_duration.as_millis() as f64 / 1e3
    );
}
