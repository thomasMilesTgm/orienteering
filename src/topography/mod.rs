//! # Topography
//!
//! Defines the topography of the world, i.e. the shape of terrain, along with biomes, structures,
//! features, etc.

use crate::utils::of32;
use nalgebra::{Point2, Vector2};
use vector_field::VectorField;

pub mod vector_field;
pub mod vector_function;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WorldMap {
    field: FieldTree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldID(pub usize);

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldTree {
    nodes: Vec<FieldNode>,
}

impl FieldTree {
    pub fn add_node<T: Into<FieldNode>>(&mut self, node: T) -> FieldID {
        let id = FieldID(self.nodes.len());
        self.nodes.push(node.into());
        id
    }

    pub fn make_child<T: Into<FieldNode>>(&mut self, parent: FieldID, field: T) -> FieldID {
        let child_id = self.add_node(field.into());
        self[parent].children.push(child_id);
        child_id
    }

    pub fn field_vector(&self, at_point: Point2<of32>) -> Vector2<f32> {
        todo!()
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
}

impl From<VectorField> for FieldNode {
    fn from(field: VectorField) -> Self {
        Self {
            field,
            children: Default::default(),
        }
    }
}

/// A rectangular area that is
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OrdArea {
    pub min: Point2<of32>,
    pub max: Point2<of32>,
}
