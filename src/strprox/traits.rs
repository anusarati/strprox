use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, Mul, Shr, ShrAssign, Sub, Shl},
};

// Rust does not have good support for generic arithmetic mixed with literals

/// Used to access unsigned primitive interface from generic types
///
/// u64 is the maximum supported type for representing the size of strings in the tree
pub trait StringSizeType: NarrowableIntermediate + Into<u64> + From<u8> {
    /// Type wider than `Self`
    type Wide: NarrowableIntermediate<Narrow = Self>;
    /// Converts `self` to a wider type
    fn into_wide(self) -> Self::Wide;
    /// Converts `self` to usize, possibly truncating
    fn into_usize(self) -> usize;
    /// Converts `value` to `Self`, possibly truncating
    fn from_usize(value: usize) -> Self;
    ///// Converts `value` to `Self`, possibly truncating
    //fn from(value: u128) -> Self;
}
pub trait NarrowableIntermediate:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Shl<Output = Self>
    + Clone
    + Ord
    + PartialOrd
    + Sized
    + Display
    + Debug
{
    /// Type at most as wide as `Self`
    type Narrow;
    /// Converts `self` into a narrower type, possibly truncating
    fn into_narrow(self) -> Self::Narrow;
}
macro_rules! impl_string_size_type {
    ($($narrow:ty)*; $($wide:ty)*) => {
            $(
                impl StringSizeType for $narrow {
                    type Wide = $wide;
                    fn into_wide(self) -> $wide {
                        self as $wide
                    }
                    fn into_usize(self) -> usize {
                        self as usize
                    }
                    fn from_usize(value: usize) -> $narrow {
                        value as $narrow
                    }
                    /*
                    fn from(value: u128) -> $narrow {
                        value as $narrow
                    }
                    */
                }
                impl NarrowableIntermediate for $wide {
                    type Narrow = $narrow;
                    fn into_narrow(self) -> $narrow {
                        self as $narrow
                    }
                }
            )*
        }
    }
// u8 can be widened to u16, u16 to 32, etc.
impl_string_size_type!( u8 u16 u32 u64; u16 u32 u64 u128 );

impl NarrowableIntermediate for u8 {
    type Narrow = u8;
    fn into_narrow(self) -> u8 {
        self
    }
}
