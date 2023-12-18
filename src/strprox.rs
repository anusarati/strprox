pub mod hs_tree;
mod gats;
mod suffix_trie;
use hs_tree::HSTree;
pub use hs_tree::MeasuredString;
pub use hs_tree::Rankings;

pub type StringSearcher<'a> = HSTree<'a>;