use std::io::{prelude::*, BufWriter};
use std::path::Path;
use std::{collections::HashSet, fs::File};

use fst::SetBuilder;
use rand::{
    distributions::{Alphanumeric, DistString, Distribution, Uniform},
    thread_rng,
};

const N_STRINGS: usize = 2e6 as usize;
const MIN_LEN: usize = 30;
const MAX_LEN: usize = 40;

/// Creates a file with `N_STRINGS` random strings with lengths in [MIN_LEN, MAX_LEN]
fn main() {
    let out_path = Path::new("../noise.txt");
    let mut file = File::create(out_path).expect("Couldn't open noise.txt for writing");

    let mut rng = thread_rng();
    let len_distribution = Uniform::new_inclusive(MIN_LEN, MAX_LEN);
    let mut strings = HashSet::<String>::new();
    while strings.len() < N_STRINGS {
        let string_len = len_distribution.sample(&mut rng);
        let rand_string = Alphanumeric.sample_string(&mut rng, string_len);
        strings.insert(rand_string);
    }
    for string in &strings {
        writeln!(&mut file, "{}", string).expect("Couldn't write string to noise.txt");
    }

    // save a FST for the noise dataset to a file
    let mut strings: Vec<_> = strings.into_iter().collect();
    strings.sort();

    let fst_path = Path::new("../noise.fst");

    let fst_writer = BufWriter::new(File::create(fst_path).unwrap());
    let mut builder = SetBuilder::new(fst_writer).unwrap();
    for string in strings {
        builder.insert(string).expect("Strings should be sorted");
    }
    builder.finish().unwrap();

}
