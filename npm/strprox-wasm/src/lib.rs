mod utils;

use wasm_bindgen::prelude::*;
use strprox::strprox::Autocompleter;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, strprox-wasm!");
}
