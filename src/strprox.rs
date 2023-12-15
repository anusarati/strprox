pub mod hs_tree;
mod gats;
use hs_tree::HSTree;
pub use hs_tree::MeasuredString;
pub use hs_tree::Rankings;

pub type StringSearcher = HSTree;