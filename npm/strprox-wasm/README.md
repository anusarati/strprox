# strprox

Autocompletion library [ported from Rust](https://github.com/anusarati/strprox) using `wasm-pack`.

## Example

Typescript
```typescript
import { Autocompleter, MeasuredPrefix } from "strprox";

let source = ["success", "foobar", "succeed"];
let autocompleter = new Autocompleter(source);

let query = "luck";
let requested = 2;
let results = autocompleter.autocomplete(query, requested);

for (const result of results) {
    console.log(result.string, result.prefix_distance);
    // Currently each result has its memory stored in `wasm-bindgen`'s heap,
    // so memory needs to manually be freed from each result using the `.free()` method.
    result.free();
}
// Output:
// succeed 2
// success 2
```