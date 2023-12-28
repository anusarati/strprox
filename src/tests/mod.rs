use std::{
    cmp::min,
    io::Write,
    time::{Duration, Instant},
};

use rand::{
    distributions::{Distribution, Uniform},
    seq::SliceRandom,
    Rng,
};

use crate::{
    levenshtein::{prefix_edit_distance, random_edits, unindexed_autocomplete},
    strprox::Autocompleter,
    MeasuredPrefix, TreeString,
};

/// Returns whether any MeasuredPrefix in `measures` has the `expected` string
fn contains_string(measures: &Vec<MeasuredPrefix>, expected: &str) -> bool {
    measures.iter().any(|measure| measure.string == expected)
}

#[test]
/// Example input from the paper on META (see the citations)
fn meta_paper_example() {
    let source: Vec<_> = vec!["soho", "solid", "solo", "solve", "soon", "throw"]
        .into_iter()
        .map(Into::into)
        .collect();
    let autocompleter = Autocompleter::<u8>::new(source.len(), source);
    let result = autocompleter.autocomplete("ssol", 3);
    for measure in &result {
        println!("{:#?}", measure);
    }
    assert!(contains_string(&result, "solid"));
    assert!(contains_string(&result, "solo"));
    assert!(contains_string(&result, "solve"));
}

#[test]
/// Tests that autocomplete can return exact associated categories
fn two_categories() {
    let source: Vec<_> = vec![
        "success",
        "successor",
        "successive",
        "decrement",
        "decrease",
        "decreasing",
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    let autocompleter = Autocompleter::<u8>::new(source.len(), source.clone());
    let query = "zucc";
    let result = autocompleter.autocomplete(query, 3);
    println!("{}\n", query);
    for measure in &result {
        println!("{:#?}", measure);
    }
    let _ = std::io::stdout().flush();

    assert!(contains_string(&result, "success"));
    assert!(contains_string(&result, "successor"));
    assert!(contains_string(&result, "successive"));
    assert_eq!(result, unindexed_autocomplete("zucc", 3, &source));

    let query = "deck";
    let result = autocompleter.autocomplete("deck", 3);
    println!("{}\n", query);
    for measure in &result {
        println!("{:#?}", measure);
    }
    assert!(contains_string(&result, "decrement"));
    assert!(contains_string(&result, "decrease"));
    assert!(contains_string(&result, "decreasing"));

    assert_eq!(result, unindexed_autocomplete("deck", 3, &source));
}

#[test]
/// The example in the README
fn example() {
    let source: Vec<_> = vec![
        "success",
        "successive",
        "successor",
        "decrease",
        "decreasing",
        "decrement",
    ]
    .into_iter()
    .map(|k| k.into())
    .collect();
    let autocompleter = Autocompleter::new(source.len(), source);
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
    let source: Vec<_> = vec!["oobf", "fbor", "bobf"]
        .into_iter()
        .map(|k| k.into())
        .collect();
    let autocompleter = Autocompleter::<u8>::new(source.len(), source);
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
    let source: Vec<TreeString> = WORDS.lines().map(|k| k.into()).collect();
    let time = Instant::now();
    let autocompleter = Autocompleter::<u8>::new(source.len(), source);
    println!("Indexing took: {:#?}", time.elapsed());
    let requested = 10;
    let result = autocompleter.autocomplete("abandonned", requested);
    assert_eq!(result.len(), requested);

    for measure in &result {
        println!("{:#?}", measure);
    }
    assert!(contains_string(&result, "abandoned"));
}

#[test]
/// Tests for error-tolerant autocompletion against a large database
fn large_database_autocomplete() {
    let source: Vec<TreeString> = WORDS.lines().map(|k| k.into()).collect();
    let autocompleter = Autocompleter::<u8>::new(source.len(), source);
    let requested = 10;
    let result = autocompleter.autocomplete("oberr", requested);
    assert_eq!(result.len(), requested);

    for measure in &result {
        println!("{:#?}", measure);
    }
    assert!(contains_string(&result, "overrank"));
}

#[test]
/// Tests that the result from an empty query still has strings
fn empty_query_test() {
    let source: Vec<TreeString> = WORDS.lines().map(|k| k.into()).collect();
    println!("words {}", source.len());
    let autocompleter = Autocompleter::<u8>::new(source.len(), source);
    let result = autocompleter.autocomplete("", 1);
    assert_ne!(result.len(), 0);
}

#[test]
/// Tests that prefix edit distances are within the number of edits made to strings from a database
/// using 1000 random data points
///
/// Simultaneously tests that the prefix edit distances are correct
fn large_database_bounded_peds() {
    let source: Vec<TreeString> = WORDS.lines().map(|k| k.into()).collect();
    let autocompleter = Autocompleter::<u8>::new(source.len(), source.clone());
    let mut rng = rand::thread_rng();
    let mut total_duration = Duration::new(0, 0);
    const ITERATIONS: usize = 1e3 as usize;
    for _i in 0..ITERATIONS {
        let string = source.choose(&mut rng).unwrap();
        let edits_distribution = Uniform::new(0, 5);
        let edits = edits_distribution.sample(&mut rng);
        let edited_string = random_edits(&string, edits);

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
        total_duration.as_millis() as f64 / ITERATIONS as f64
    );
}
