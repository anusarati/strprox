pub mod traits;
pub mod hs_tree;
mod gats;
use hs_tree::HSTree;
pub use hs_tree::MeasuredString;
pub use hs_tree::Rankings;

use self::traits::StringSizeType;

pub type StringSearcher<'a, U: StringSizeType> = HSTree<'a, U>;