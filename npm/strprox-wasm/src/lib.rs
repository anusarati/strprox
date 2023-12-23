mod utils;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub struct Autocompleter {
    base: strprox::Autocompleter<'static, u8, u32>,
}

impl Autocompleter {
    /// Allocates strings from source
    #[wasm_bindgen(constructor)]
    pub fn new(source: &[&str]) -> Self {
        // give all the strings static lifetime
        let mut strings = Vec::<&'static str>::with_capacity(source.len());
        for &string in source {
            strings.push(Box::leak(Box::new(string)));
        }
        Self {
            base: strprox::Autocompleter::<'static, u8, 32>::new(strings.as_slice())
        }
    }
    /// Returns the best `requested` strings for autocompletion as a Javascript array of MeasuredPrefix
    #[wasm_bindgen]
    pub fn autocomplete(&self, query: &str, requested: usize) -> js_sys::Array {
        let result = self.base.autocomplete(query, requested);
        let js_result = js_sys::Array::with_capacity(result.len());
    }
}
