use std::{
    fs,
    io::Write,
    time::{Duration, Instant}
};

use fst::Set;
use rand::thread_rng;
use yoke::Yoke;

use crate::{
    levenshtein::{prefix_edit_distance, sample_edited_string, unindexed_autocomplete},
    strprox::FstAutocompleter,
    strprox::MetaAutocompleter,
    Autocompleter, MeasuredPrefix,
};

type YokedMetaAutocompleter = Yoke<MetaAutocompleter<'static>, Vec<String>>;

/// Returns whether any MeasuredPrefix in `measures` has the `expected` string
fn contains_string(measures: &Vec<MeasuredPrefix>, expected: &str) -> bool {
    measures.iter().any(|measure| measure.string == expected)
}

// Words from https://github.com/dwyl/english-words/blob/master/words.txt
const WORDS: &str = include_str!("words.txt");

#[generic_tests::define]
mod generic {
    use super::*;
    use crate::{levenshtein, Autocompleter};

    #[test]
    /// Example input from the paper on META (see the citations)
    fn meta_paper_example<A>()
    where
        A: Autocompleter,
    {
        let source: Vec<_> = vec!["soho", "solid", "solo", "solve", "soon", "throw"];
        let autocompleter = A::from_strings(&source);
        let result = autocompleter.autocomplete("ssol", 3);
        for measure in &result {
            println!("{:#?}", measure);
        }
        // these are the only strings with PEDs of 1
        assert!(contains_string(&result, "solid"));
        assert!(contains_string(&result, "solo"));
        assert!(contains_string(&result, "solve"));
    }
    #[test]
    /// Tests that autocomplete can return exact associated categories
    fn two_categories<A>()
    where
        A: Autocompleter,
    {
        let source: Vec<_> = vec![
            "success",
            "successor",
            "successive",
            "decrement",
            "decrease",
            "decreasing",
        ];
        let autocompleter = A::from_strings(&source);
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

        let cows: Vec<_> = source.iter().map(|&s| s.into()).collect();
        assert_eq!(result, unindexed_autocomplete("zucc", 3, &cows));

        let query = "deck";
        let result = autocompleter.autocomplete("deck", 3);
        println!("{}\n", query);
        for measure in &result {
            println!("{:#?}", measure);
        }
        assert!(contains_string(&result, "decrement"));
        assert!(contains_string(&result, "decrease"));
        assert!(contains_string(&result, "decreasing"));

        assert_eq!(result, unindexed_autocomplete("deck", 3, &cows));
    }

    #[test]
    /// The example in the README
    fn example<A>()
    where
        A: Autocompleter,
    {
        let source: Vec<_> = vec![
            "success",
            "successive",
            "successor",
            "decrease",
            "decreasing",
            "decrement",
        ];
        let autocompleter = A::from_strings(&source);
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
    fn insertion_ped<A>()
    where
        A: Autocompleter,
    {
        let query = "foob";
        // PEDs: [1, 2, 2]
        let source: Vec<_> = vec!["oobf", "fbor", "bobf"]
            .into_iter()
            .map(|k| k.into())
            .collect();
        let autocompleter = A::from_strings(&source);
        let result = autocompleter.autocomplete(query, 1);
        for measure in &result {
            println!("{:#?}", measure);
        }
        assert_eq!(result[0].string, "oobf");
    }

    #[test]
    /// Tests for correction of a misspelling against a large database
    fn words_misspelling<A>()
    where
        A: Autocompleter,
    {
        let source: Vec<_> = WORDS.lines().collect();
        let time = Instant::now();
        let autocompleter = A::from_strings(&source);
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
    fn words_autocomplete<A>()
    where
        A: Autocompleter,
    {
        let source: Vec<_> = WORDS.lines().collect();
        let autocompleter = A::from_strings(&source);
        let requested = 10;
        let query = "oberr";
        let result = autocompleter.autocomplete(query, requested);
        assert_eq!(result.len(), requested);

        /// Max PED of any top-10 strings from the dataset against the query "oberr"
        const MAX_PED: usize = 1;

        for measure in &result {
            println!("{:#?}", measure);
            assert!(measure.prefix_distance <= MAX_PED);
            assert_eq!(
                measure.prefix_distance,
                levenshtein::prefix_edit_distance(query, measure.string.as_str())
            );
        }

        // this requires increasing the requested number for the fst implementation,
        // because there are 10 strings that start with "aberr", which also have PED of 1
        //assert!(contains_string(&result, "overrank"));
    }

    #[test]
    /// Tests for error-tolerant autocompletion against a large database
    fn words_long_query<A>()
    where
        A: Autocompleter,
    {
        let source: Vec<_> = WORDS.lines().collect();
        let autocompleter = A::from_strings(&source);
        let requested = 3;
        println!("begin");
        let result = autocompleter.autocomplete("asfdasdvSDVASDFEWWEFWDASDAS", requested);
        assert_eq!(result.len(), requested);

        for measure in &result {
            println!("{:#?}", measure);
        }
    }

    #[test]
    /// Tests that any result has a PED under the given threshold
    fn words_threshold_topk<A>()
    where
        A: Autocompleter,
    {
        let source: Vec<_> = WORDS.lines().collect();
        let autocompleter = A::from_strings(&source);
        let requested = 3;
        println!("begin");
        let query = "asfdasdvSDVASDFEWWEFWDASDAS";
        let result = autocompleter.threshold_topk(query, requested, 5);
        dbg!(&result);
        assert_eq!(
            result.len(),
            0,
            "PEDs of results above 20 but threshold is 10"
        );
    }

    #[test]
    /// Tests for error-tolerant autocompletion against a large database
    fn words_long_query_exist<A>()
    where
        A: Autocompleter,
    {
        let source: Vec<_> = WORDS.lines().collect();
        let autocompleter = A::from_strings(&source);
        let requested = 3;
        println!("begin"); // nonsyntactically
        let result = autocompleter.autocomplete("nonsyntacticallz", requested);
        assert_eq!(result.len(), requested);

        for measure in &result {
            println!("{:#?}", measure);
        }
    }

    #[test]
    /// Tests that the result from an empty query still has strings
    fn empty_query_test<A>()
    where
        A: Autocompleter,
    {
        let source: Vec<_> = WORDS.lines().collect();
        println!("words {}", source.len());
        let autocompleter = A::from_strings(&source);
        let result = autocompleter.autocomplete("", 1);
        assert_ne!(result.len(), 0);
    }

    #[test]
    /// Tests that prefix edit distances are within the number of edits made to strings from a database
    /// using 1000 random data points
    ///
    /// Simultaneously tests that the prefix edit distances are correct
    fn words_bounded_peds<A>()
    where
        A: Autocompleter,
    {
        let source: Vec<_> = WORDS.lines().collect();
        let autocompleter = A::from_strings(&source);
        let mut rng = rand::thread_rng();
        let mut total_duration = Duration::new(0, 0);
        const ITERATIONS: usize = 1e3 as usize;
        for _i in 0..ITERATIONS {
            let (string, edited_string, edits) = sample_edited_string(&source, &mut rng);

            dbg!(&string, &edited_string);

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

    #[instantiate_tests(<YokedMetaAutocompleter>)]
    mod meta {}
    #[instantiate_tests(<FstAutocompleter>)]
    mod fst {}
}

// ideally this would use the #[bench] attribute but it's unstable
#[ignore]
#[test]
/// Benchmark the unindexed autocomplete against the request sampling used in the words_bounded_peds test
fn bench_unindexed() {
    let source: Vec<_> = WORDS.lines().collect();
    let cows: Vec<_> = source.iter().map(|&s| s.into()).collect();
    let mut rng = rand::thread_rng();
    let mut total_duration = Duration::new(0, 0);
    const ITERATIONS: usize = 1e3 as usize;
    for _i in 0..ITERATIONS {
        let (_, edited_string, _) = sample_edited_string(&source, &mut rng);
        let time = Instant::now();
        let result = &unindexed_autocomplete(edited_string.as_str(), 1, &cows)[0];
        total_duration += time.elapsed();
        dbg!(result);
    }
    println!(
        "Average time per query: {} ms",
        total_duration.as_millis() as f64 / ITERATIONS as f64
    );
}

#[ignore]
#[test]
/// Check the performance of the autocomplete methods against the noise dataset
fn bench_noise() {
    const TEXT: &str = include_str!("noise.txt");
    let mut source: Vec<&str> = TEXT.lines().collect();
    let cows: Vec<_> = source.iter().map(|&s| s.into()).collect();
    let mut start = Instant::now();
    source.sort();

    println!("Testing without a max threshold");

    let fst_autocomp = FstAutocompleter::from_strings(&source);
    println!("FST indexing took {} ms", start.elapsed().as_millis());

    let requested = 10;
    let query = &"z".repeat(35);

    start = Instant::now();
    let mut result = fst_autocomp.autocomplete(query, requested);
    println!("Autocomplete took {} ms", start.elapsed().as_millis());

    for measure in result {
        println!("{}", measure);
    }

    start = Instant::now();
    let fst_data = fs::read("src/tests/noise.fst").expect("noise.fst should exist");
    let fst_autocomp = FstAutocompleter::new(Set::new(fst_data).unwrap().into_fst());
    println!(
        "Construction from file took {} ms",
        start.elapsed().as_millis()
    );

    start = Instant::now();
    result = unindexed_autocomplete(query, requested, &cows);
    println!(
        "Unindexed autocomplete took {} ms",
        start.elapsed().as_millis()
    );

    for measure in result {
        println!("{}", measure);
    }

    // my implementation of META is too slow for the zzz query

    start = Instant::now();
    let meta_autocomp = MetaAutocompleter::new(cows.len(), cows);
    println!("META indexing took {} ms", start.elapsed().as_millis());

    println!("\nTesting with a maximum threshold");

    const ITERATIONS: usize = 1e2 as usize;
    // test random queries with a PED of at most `MAX_THRESHOLD`
    const MAX_THRESHOLD: usize = 4;
    let mut rng = thread_rng();
    for _i in 0..ITERATIONS {
        let edited_string = sample_edited_string(&source, &mut rng).1;
        let query = edited_string.as_str();

        start = Instant::now();
        let result = fst_autocomp.threshold_topk(query, requested, MAX_THRESHOLD);
        println!("Fst autocomplete took {} ms", start.elapsed().as_millis());
        dbg!(result);

        start = Instant::now();
        let result = meta_autocomp.threshold_topk(query, requested, MAX_THRESHOLD);
        println!("META autocomplete took {} ms", start.elapsed().as_millis());
        dbg!(result);
    }
}
