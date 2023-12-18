use std::collections::HashMap;
// using generic associated types for higher-kinded type instantiation
// https://rust-lang.github.io/rfcs/1598-generic_associated_types.html
pub trait IterableMap {
    type K;
    type V;
    type Iter<'a>: Iterator<Item = (&'a Self::K, &'a Self::V)>
    where
        Self: 'a;
    fn iter<'a>(&self) -> Self::Iter<'a> {
        self.iter()
    }
}
impl<_K, _V> IterableMap for HashMap<_K, _V> {
    type K = _K;
    type V = _V;
    type Iter<'a> = std::collections::hash_map::Iter<'a, Self::K, Self::V> where Self::K: 'a, Self::V: 'a;
}
pub trait MapGAT {
    type Map<K, V>: IterableMap<K = K, V = V>;
}

#[derive(Clone)]
pub enum HashMapGAT {}

impl MapGAT for HashMapGAT {
    type Map<K, V> = HashMap<K, V>;
}

pub trait ArrayGAT {
    type Array<T>;
}

pub enum VecArrayGAT {}

impl ArrayGAT for VecArrayGAT {
    type Array<T> = Vec<T>;
}
