# strprox

strprox is a library that currently allows for top-k string autocompletion.

This is intended to implement the top-k "matching-based framework for error-tolerant autocompletion" algorithm from the paper by Deng et al. (see the [Citations](#citations))

## Table of Contents
- [Example](#example)
- [Notes](#notes)
- [Citations](#citations)
- [License](#license)
- [Contributions](#contrib)

## [Example](#example)

```rust
use strprox::Autocompleter;

fn main() {
    // Here's the data that the autocompleter uses
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

    // Retrieve the 3 best strings for autocompletion
    let result = autocompleter.autocomplete(query, 3);

    // (string: success, PED: 2)
    // (string: successive, PED: 2)
    // (string: successor, PED: 2)
    for measured_prefix in &result {
        println!("{}", measured_prefix);
    }

    // Collect the strings from the resulting string and prefix edit distance combination
    let result_strings: Vec<&str> = result
        .iter()
        .map(|measured_prefix| measured_prefix.string.as_str())
        .collect();

    assert_eq!(result_strings, vec!["success", "successive", "successor"]);
}
```

## [Notes](#notes)

`Autocompleter` is currently a generic struct that limits the length of referenced strings to `u8::MAX` and maximum number of referenced strings to `u32::MAX` by default for space efficiency. A macro can be added later to instantiate it for other unsigned types.

Some of the tests require a file named [`words.txt`](https://github.com/dwyl/english-words/blob/master/words.txt) file (highlighted link) in `src/tests`.

## [Citations](#citations)
```bibtex
@article{10.14778/2977797.2977808,
author = {Deng, Dong and Li, Guoliang and Wen, He and Jagadish, H. V. and Feng, Jianhua},
title = {META: An Efficient Matching-Based Method for Error-Tolerant Autocompletion},
year = {2016},
issue_date = {June 2016},
publisher = {VLDB Endowment},
volume = {9},
number = {10},
issn = {2150-8097},
url = {https://doi.org/10.14778/2977797.2977808},
doi = {10.14778/2977797.2977808},
journal = {Proc. VLDB Endow.},
month = {jun},
pages = {828–839},
numpages = {12}
}
```

## [License](#license)
Dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE). You may choose either license.

### [Contributions](#contrib)
Contributions to this project are likewise understood to be dual-licensed under MIT and Apache-2.0.
