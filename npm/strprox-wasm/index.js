import { initSync } from "./pkg/strprox.js";
import wasm from "./pkg/strprox_bg.wasm";

// https://github.com/rollup/plugins/tree/master/packages/wasm
initSync(await wasm());

export { Autocompleter, MeasuredPrefix, MeasuredString } from "./pkg/strprox.js";