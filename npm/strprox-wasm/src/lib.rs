mod utils;

use gloo_utils::format::JsValueSerdeExt;
use wasm_bindgen::prelude::*;

// Typescript binding for MeasuredPrefix
// currently cannot include_str!("measured_prefix.ts")
// https://github.com/rustwasm/wasm-bindgen/issues/2828
#[wasm_bindgen(typescript_custom_section)]
const MEASURED_PREFIX_TS: &'static str = r#"export type MeasuredPrefix = {
    string: string;
    prefix_distance: number;
};"#;

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
    /// Returns the best `requested` number of strings for autocompleting `query`
    /// as a Javascript array of MeasuredPrefix
    pub fn autocomplete(&self, query: &str, requested: usize) -> js_sys::Array {
        let result = self.base.autocomplete(query, requested);
        let js_result = js_sys::Array::new();
        for measure in result {
            js_result.push(&JsValue::from_serde(&measure).unwrap());
        }
        js_result
    }
}
