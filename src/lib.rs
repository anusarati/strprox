pub mod strprox;

#[doc(inline)]
pub use strprox::*;

#[cfg(test)]
mod levenshtein;
#[cfg(test)]
mod tests;
