//! # Topography
//!
//! Defines the topography of the world, i.e. the shape of terrain, along with biomes, structures,
//! features, etc.

use crate::{
    proc_gen::{MapSeed, ProceduralValue},
    utils::*,
};
use nalgebra::{Point2, Vector2};
use vector_field::{Field, StationaryField, VectorField};

pub mod vector_field;
pub mod vector_function;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WorldMap {
    field: FieldTree,
}

impl WorldMap {
    pub fn new(seed: MapSeed) -> Self {
        let mut rng = seed.into_small_rng();

        // The base field is stationary
        let field = VectorField::new(StationaryField::from_rng(&mut rng));

        WorldMap {
            field: FieldTree::new(field),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldID(usize);

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldTree {
    nodes: Vec<FieldNode>,
}

impl FieldTree {
    pub fn new<T: Into<FieldNode>>(field: T) -> Self {
        FieldTree {
            nodes: vec![field.into()],
        }
    }

    pub fn push_node<T: Into<FieldNode>>(&mut self, node: T) -> FieldID {
        let id = FieldID(self.nodes.len());
        self.nodes.push(node.into());
        id
    }

    pub fn make_child<T: Into<FieldNode>>(
        &mut self,
        parent: FieldID,
        field: T,
        area: AreaOF,
    ) -> FieldID {
        let mut field: FieldNode = field.into();
        field.influence = area;
        let child_id = self.push_node(field);

        let parent = &mut self[parent];
        parent.children.push(child_id);

        child_id
    }

    pub fn field_vector(&self, pt: Point2<of32>) -> Vector2<of32> {
        self.root_field().field_vector(self, pt)
    }

    pub fn root_field(&self) -> &FieldNode {
        &self.nodes[0]
    }

    pub fn root_id(&self) -> FieldID {
        FieldID(0)
    }
}

impl std::ops::Index<FieldID> for FieldTree {
    type Output = FieldNode;

    fn index(&self, index: FieldID) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl std::ops::IndexMut<FieldID> for FieldTree {
    fn index_mut(&mut self, index: FieldID) -> &mut Self::Output {
        &mut self.nodes[index.0]
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldNode {
    /// The Vector field applied by this node within it's area of influence.
    field: VectorField,

    /// Fields that overlay onto this one
    children: Vec<FieldID>,

    /// The area of influence of this vector field
    influence: AreaOF,
}

impl FieldNode {
    /// Compute the vector direction of [`Self::field`], recursively accumulating the effect of
    /// child fields.
    pub fn field_vector<D>(&self, data: &D, pt: Point2<of32>) -> Vector2<of32>
    where
        D: std::ops::Index<FieldID, Output = FieldNode>,
    {
        let this_area = self.influence.area();
        let mut vector = self.field.v_xy(pt);

        if let Some(child) = self.child_at_point(data, pt) {
            let child = &data[*child];
            let child_area = child.influence.area();
            let child_weight = child_area / this_area;

            let mut child_vector = child.field_vector(data, pt);

            // XXX: Nalgebra doens't like ordered float
            child_vector.iter_mut().for_each(|v| *v *= child_weight);

            vector += child_vector;
        }

        vector
    }

    pub fn child_at_point<D>(&self, data: &D, pt: Point2<of32>) -> Option<&FieldID>
    where
        D: std::ops::Index<FieldID, Output = FieldNode>,
    {
        self.children
            .iter()
            .find(|child| data[**child].influence.contains(pt))
    }
}

impl From<VectorField> for FieldNode {
    fn from(field: VectorField) -> Self {
        Self {
            field,
            influence: AreaOF::infinity(),
            children: Default::default(),
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RangeOF<C> {
    min: of32,
    max: of32,
    phantom: std::marker::PhantomData<C>,
}

impl<C> RangeOF<C> {
    pub fn new(min: of32, max: of32) -> Self {
        RangeOF {
            min,
            max,
            phantom: std::marker::PhantomData,
        }
    }
    pub fn contains(&self, value: of32) -> bool {
        value >= self.min && value <= self.max
    }
    pub fn is_empty(&self) -> bool {
        self.min > self.max
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct X;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Y;

pub type YRangeOF = RangeOF<Y>;
pub type XRangeOF = RangeOF<X>;

/// A rectangular area that is
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AreaOF {
    pub min: Point2<of32>,
    pub max: Point2<of32>,
}

impl AreaOF {
    pub fn zero() -> Self {
        AreaOF {
            min: Point2::origin(),
            max: Point2::origin(),
        }
    }
    pub fn infinity() -> Self {
        let neg: of32 = f32::NEG_INFINITY.into();
        let pos: of32 = f32::NEG_INFINITY.into();
        AreaOF {
            min: Point2::new(neg, neg),
            max: Point2::new(pos, pos),
        }
    }
    pub fn area(&self) -> of32 {
        self.width() * self.height()
    }

    pub fn width(&self) -> of32 {
        self.max.x - self.min.x
    }

    pub fn height(&self) -> of32 {
        self.max.y - self.min.y
    }
    pub fn contains(&self, pt: Point2<of32>) -> bool {
        pt.x >= self.min.x && pt.x <= self.max.x && pt.y >= self.min.y && pt.y <= self.max.y
    }

    pub fn xy_ranges(&self) -> (XRangeOF, YRangeOF) {
        (self.x_range(), self.y_range())
    }
    pub fn y_range(&self) -> YRangeOF {
        RangeOF::new(self.min.y, self.max.y)
    }
    pub fn x_range(&self) -> XRangeOF {
        RangeOF::new(self.min.x, self.max.x)
    }
}
