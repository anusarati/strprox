use crate::{strprox, levenshtein};
use strprox::MeasuredPrefix;
use wasm_bindgen::prelude::*;


#[wasm_bindgen]
pub struct Autocompleter {
    base: strprox::Autocompleter<'static, u8, u32>,
}

#[wasm_bindgen]
impl Autocompleter {
    /// Allocates strings from a Javascript array of strings `source`
    /// and uses them as reference for autocompletion
    #[wasm_bindgen(constructor)]
    pub fn new(source: js_sys::Array) -> Autocompleter {
        // copy all the strings to Rust
        let mut strings: Vec<String> = Vec::<String>::with_capacity(source.length() as usize);
        for value in source {
            if let Some(string) = value.as_string() {
                strings.push(string);
            }
        }
        // conserve memory (to optimize with Node startup snapshot)
        // this is operation is unfortunately duplicated in the construction of strprox::Autocompleter,
        // which does not assume the source is sorted and deduped, but using a snapshot should mitigate
        // this drawback
        strings.sort_unstable();
        strings.dedup();

        // give all the strings static lifetime
        let mut static_string_refs = Vec::<&'static str>::with_capacity(strings.len());
        for string in strings {
            static_string_refs.push(Box::leak(Box::new(string)).as_str());
        }

        let slice: &'static [&'static str] = Box::leak(Box::new(static_string_refs)).as_slice();
        let base = strprox::Autocompleter::<'static, u8, u32>::new(slice);
        Self { base }
    }
    /// Returns the best `requested` number of strings with their prefix edit distances for autocompleting `query`
    pub fn autocomplete(&self, query: &str, requested: usize) -> Vec<MeasuredPrefix> {
        self.base.autocomplete(query, requested)
    }
}

#[wasm_bindgen]
/// Returns the `requested` number of strings with the best prefix edit distance from the `query`
/// without using an index
pub fn unindexed_autocomplete(
    query: &str,
    requested: usize,
    source: js_sys::Array,
) -> Vec<MeasuredPrefix> {
    let mut internal_strings = Vec::<String>::with_capacity(source.length() as usize);
    for value in source {
        if let Some(string) = value.as_string() {
            internal_strings.push(string);
        }
    }
    let strings: Vec<&str> = internal_strings
        .iter()
        .map(|string| string.as_str())
        .collect();
    levenshtein::unindexed_autocomplete(query, requested, &strings)
}
