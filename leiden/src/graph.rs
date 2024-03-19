use num_traits::bounds::Bounded;
use num_traits::cast::AsPrimitive;
use num_traits::identities::{One, Zero};
use std::convert::{From, TryFrom};
use std::iter::{repeat, FlatMap, Iterator, Repeat, Zip};
use std::ops::{Add, AddAssign};
use std::slice::Iter;

pub trait IndexTrait:
    Add<Output = Self>
    + AddAssign
    + AsPrimitive<usize>
    + Bounded
    + Clone
    + Copy
    + One
    + PartialEq
    + PartialOrd
    + TryFrom<usize>
    + Zero
where
    Self: std::marker::Sized,
{
}

impl IndexTrait for usize {}
impl IndexTrait for u32 {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct Index<Ix>(Ix)
where
    Ix: IndexTrait;

impl<Ix> Index<Ix>
where
    Ix: IndexTrait,
{
    pub fn index(&self) -> Ix {
        self.0
    }
    pub fn end() -> Self {
        Index(Ix::max_value())
    }
}

impl<Ix> From<Ix> for Index<Ix>
where
    Ix: IndexTrait,
{
    fn from(ix: Ix) -> Self {
        Index(ix)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Edge<W, NodeIx>
where
    W: Clone,
    NodeIx: IndexTrait,
{
    source: Index<NodeIx>,
    target: Index<NodeIx>,
    weight: W,
}

impl<W, NodeIx> Edge<W, NodeIx>
where
    W: Clone,
    NodeIx: IndexTrait,
{
    pub fn source(&self) -> Index<NodeIx> {
        self.source
    }
    pub fn target(&self) -> Index<NodeIx> {
        self.target
    }
    pub fn weight(&self) -> W {
        self.weight.clone()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DiEdge<W, NodeIx = usize>
where
    W: Clone,
    NodeIx: IndexTrait,
{
    target: Index<NodeIx>,
    pub(crate) weight: W,
}

pub struct Edges<'a, W, NodeIx = usize>
where
    W: Clone,
    NodeIx: IndexTrait,
{
    source: Index<NodeIx>,
    iter: Iter<'a, DiEdge<W, NodeIx>>,
}

impl<'a, W, NodeIx> Iterator for Edges<'a, W, NodeIx>
where
    W: Clone,
    NodeIx: IndexTrait,
{
    type Item = Edge<&'a W, NodeIx>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|DiEdge { target, weight }| Edge {
            source: self.source,
            target: *target,
            weight,
        })
    }
}

fn edge_refs_mapper<W, NodeIx>(
    (src, edges): (Index<NodeIx>, &Vec<DiEdge<W, NodeIx>>),
) -> Zip<Repeat<Index<NodeIx>>, Iter<DiEdge<W, NodeIx>>>
where
    W: Clone,
    NodeIx: IndexTrait,
{
    repeat(src).zip(edges.iter())
}

type EdgeRefsMapper<W, NodeIx> =
    fn((Index<NodeIx>, &Vec<DiEdge<W, NodeIx>>)) -> Zip<Repeat<Index<NodeIx>>, Iter<DiEdge<W, NodeIx>>>;
type IndexEdgeMapInput<'a, W, NodeIx> = Zip<Repeat<Index<NodeIx>>, Iter<'a, DiEdge<W, NodeIx>>>;
type IndexEdgeMapOutput<'a, W, NodeIx> = Zip<NodeIndices<NodeIx>, Iter<'a, Vec<DiEdge<W, NodeIx>>>>;

pub struct EdgeReferences<'a, W, NodeIx = usize>
where
    W: Add<Output = W> + Clone,
    NodeIx: IndexTrait,
{
    iter: FlatMap<IndexEdgeMapOutput<'a, W, NodeIx>, IndexEdgeMapInput<'a, W, NodeIx>, EdgeRefsMapper<W, NodeIx>>,
}

impl<'a, W, NodeIx> Iterator for EdgeReferences<'a, W, NodeIx>
where
    W: Add<Output = W> + Clone,
    NodeIx: IndexTrait,
{
    type Item = Edge<&'a W, NodeIx>;

    fn next(&mut self) -> Option<Self::Item> {
        for (source, DiEdge { target, weight }) in self.iter.by_ref() {
            if target.0 >= source.0 {
                return Some(Edge {
                    source,
                    target: *target,
                    weight,
                });
            }
        }
        None
    }
}

pub struct NodeIndices<Ix>
where
    Ix: IndexTrait,
{
    start: Ix,
    end: Ix,
}

impl<Ix> Iterator for NodeIndices<Ix>
where
    Ix: IndexTrait,
{
    type Item = Index<Ix>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let ix = self.start;
            self.start += Ix::one();
            Some(Index(ix))
        } else {
            None
        }
    }
}

pub struct UnGraph<NodeW, EdgeW, NodeIx = usize>
where
    NodeW: Add<Output = NodeW> + AddAssign + Clone + Zero,
    EdgeW: Add<Output = EdgeW> + AddAssign + Clone + Zero,
    NodeIx: IndexTrait,
    <NodeIx as TryFrom<usize>>::Error: std::fmt::Debug,
{
    pub(crate) edges: Vec<Vec<DiEdge<EdgeW, NodeIx>>>,
    node_weights: Vec<NodeW>,
    total_edges: usize,
    total_nodes: NodeIx,
}

impl<NodeW, EdgeW, NodeIx> UnGraph<NodeW, EdgeW, NodeIx>
where
    NodeW: Add<Output = NodeW> + AddAssign + Clone + Zero,
    EdgeW: Add<Output = EdgeW> + AddAssign + Clone + Zero,
    NodeIx: IndexTrait,
    <NodeIx as TryFrom<usize>>::Error: std::fmt::Debug,
{
    pub fn add_edge(&mut self, source: Index<NodeIx>, target: Index<NodeIx>, weight: EdgeW) {
        self.edges[source.0.as_()].push(DiEdge {
            target,
            weight: weight.clone(),
        });
        self.edges[target.0.as_()].push(DiEdge { target: source, weight });
        self.total_edges += 1;
    }

    pub fn add_node(&mut self, weight: NodeW) -> Index<NodeIx> {
        let index = self.total_nodes;
        self.edges.push(vec![]);
        self.node_weights.push(weight);
        self.total_nodes += NodeIx::one();
        Index(index)
    }

    pub fn edge_count(&self) -> usize {
        self.total_edges.as_()
    }

    pub fn node_count(&self) -> NodeIx {
        self.total_nodes
    }

    pub fn node_weight(&self, ix: Index<NodeIx>) -> Option<&NodeW> {
        let ix: usize = ix.0.as_();
        self.node_weights.get(ix)
    }

    pub fn node_weight_mut(&mut self, ix: Index<NodeIx>) -> Option<&mut NodeW> {
        let ix: usize = ix.0.as_();
        self.node_weights.get_mut(ix)
    }

    pub fn edges(&self, source: Index<NodeIx>) -> Edges<EdgeW, NodeIx> {
        Edges {
            source,
            iter: self.edges[source.0.as_()].iter(),
        }
    }

    pub fn node_indices(&self) -> NodeIndices<NodeIx> {
        NodeIndices {
            start: NodeIx::zero(),
            end: self.total_nodes,
        }
    }

    pub fn with_capacity(nodes: usize, _edges: usize) -> Self {
        UnGraph {
            edges: Vec::with_capacity(nodes),
            node_weights: Vec::with_capacity(nodes),
            total_edges: 0,
            total_nodes: NodeIx::zero(),
        }
    }

    pub fn edge_references(&self) -> EdgeReferences<EdgeW, NodeIx> {
        let iter = self
            .node_indices()
            .zip(self.edges.iter())
            .flat_map(edge_refs_mapper as EdgeRefsMapper<EdgeW, NodeIx>);
        EdgeReferences { iter }
    }

    pub fn filter_map<'a, F, G, N2, E2>(&'a self, mut node_map: F, mut edge_map: G) -> UnGraph<N2, E2, NodeIx>
    where
        N2: Add<Output = N2> + AddAssign + Clone + Zero,
        E2: Add<Output = E2> + AddAssign + Clone + Zero,
        F: FnMut(Index<NodeIx>, &'a NodeW) -> Option<N2>,
        G: FnMut(&'a EdgeW) -> Option<E2>,
    {
        let mut graph = UnGraph::with_capacity(0, 0);
        let mut node_index_map = vec![Index::<NodeIx>::end(); self.node_count().as_()];
        for (ix, weight) in self.node_indices().zip(self.node_weights.iter()) {
            if let Some(weight) = node_map(ix, weight) {
                node_index_map[ix.0.as_()] = graph.add_node(weight);
            }
        }
        for edge in self.edge_references() {
            let source = node_index_map[edge.source().index().as_()];
            let target = node_index_map[edge.target().index().as_()];
            if source != Index::<NodeIx>::end() && target != Index::<NodeIx>::end() {
                if let Some(weight) = edge_map(edge.weight) {
                    graph.add_edge(source, target, weight);
                }
            }
        }
        graph
    }
}
