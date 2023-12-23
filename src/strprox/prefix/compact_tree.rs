// currently unsupported
use super::*;
type UUU = u8;
type SSS = u32;
struct CompactTreeNode<'stored, UUU, SSS> {
    node: &'stored Node<UUU, SSS>,
    children: ChildrenList<'stored, UUU, SSS>,
    parent: Option<Box<CompactTreeNode<'stored, UUU, SSS>>>,
}

struct ChildrenListNode<'stored, UUU, SSS> {
    node: CompactTreeNode<'stored, UUU, SSS>,
    next: Option<Box<ChildrenListNode<'stored, UUU, SSS>>>,
}

/// Using singly-linked list to merge nodes at different levels
#[derive(Default)]
struct ChildrenList<'stored, UUU, SSS> {
    head: Option<Box<ChildrenListNode<'stored, UUU, SSS>>>,
}

/// The Compact Tree structure from META for storing active nodes
struct CompactTree<'stored, UUU, SSS> {
    root: Option<CompactTreeNode<'stored, UUU, SSS>>,
}

impl<'stored> CompactTree<'stored, UUU, SSS> {
    /// Constructs a CompactTree with the root of a trie
    fn new(trie: &'stored Trie<'stored, UUU, SSS>) -> Self {
        let root = CompactTreeNode::<'stored, UUU, SSS> {
            node: trie.root(),
            children: Default::default(),
            parent: None
        };
        Self { root: Some(root) }
    }
    fn insert(ancestor: &'stored Node<UUU, SSS>, nodes: &'stored [Node<UUU, SSS>], depth: UUU) {}
}