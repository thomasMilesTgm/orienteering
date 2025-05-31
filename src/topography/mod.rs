//! # Topography
//!
//! Defines the topography of the world, i.e. the shape of terrain, along with biomes, structures,
//! features, etc.

use crate::{
    proc_gen::{MapSeed, ProceduralValue},
    utils::of32,
};
use nalgebra::{Point2, Vector2};
use rand::rngs::SmallRng;
use vector_field::*;

pub mod vector_field;
pub mod vector_function;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WorldMap {
    pub contours: Vec<Contour>,

    pub field: FieldTree,

    pub seed: MapSeed,

    #[serde(skip)]
    rng: Option<SmallRng>,
}

impl WorldMap {
    pub fn new(seed: MapSeed) -> Self {
        let mut rng = seed.clone().into_small_rng();

        // The base field is stationary
        let field = VectorField::new(StationaryField::from_rng(&mut rng));

        WorldMap {
            field: FieldTree::new(field),
            rng: Some(rng),
            seed,
            contours: vec![],
        }
    }

    pub fn rng(&mut self) -> &mut SmallRng {
        if self.rng.is_none() {
            self.rng = Some(self.seed.clone().into_small_rng());
        }
        self.rng.as_mut().unwrap()
    }

    pub fn generate_area<T: Into<FieldType>>(&mut self, area: AreaOF, field: T) {
        let child = VectorField::new(field.into());
        let parent = self.find_parent_node(&area);
        self.field.make_child(parent, child, area);
    }

    pub fn generate_contour(&mut self, mut from: Point2<f32>, mut length: f32, z: f32) {
        let mut contour = Contour {
            z,
            ..Default::default()
        };

        const DT: f32 = 0.1;
        const CLOSE: f32 = 1.;

        let mut left_start = false;

        while length > 0. {
            if !left_start {
                left_start = contour
                    .points
                    .iter()
                    .any(|pt| (from - *pt).magnitude() > 2. * CLOSE);
            } else if (from - contour.points[0]).magnitude() < CLOSE {
                println!("Contour closed at {:?}", from);
                break;
            }

            let dr = self.field.field_vector(from) * DT;

            if dr.magnitude() < 0.01 {
                println!("Contour became stationary at {:?}", from);
                break;
            }

            from += dr;
            length -= dr.magnitude();

            contour.points.push(from);
            contour.tangents.push(dr);
        }

        self.contours.push(contour);
    }

    pub fn generate_island(&mut self, area: AreaOF) {
        let circular = CircularField::from_rng(self.rng());
        self.generate_area(area, circular);

        for child_area in area.quadrants() {
            let field = FieldType::from_rng(self.rng());
            self.generate_area(child_area, field);
        }
        // let hilly = HillyBowlField::from_rng(self.rng());
        // self.generate_area(area, hilly);

        let dx = area.width() * 0.4;

        let start = Point2::new(*area.min.x + dx, *area.min.y + dx);
        self.generate_contour(start, 10000., 0.);
    }

    fn find_parent_node(&self, area: &AreaOF) -> FieldID {
        let center = area.center();
        let mut field_id = self.field.root_id();
        while let Some(child) = self.field[field_id].child_at_point(&self.field, center) {
            field_id = *child;
        }
        field_id
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Contour {
    pub points: Vec<Point2<f32>>,
    pub tangents: Vec<Vector2<f32>>,
    pub z: f32,
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

    pub fn field_vector(&self, pt: Point2<f32>) -> Vector2<f32> {
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
    pub fn field_vector<D>(&self, data: &D, pt: Point2<f32>) -> Vector2<f32>
    where
        D: std::ops::Index<FieldID, Output = FieldNode>,
    {
        let this_area = self.influence.area();
        let mut vector = self.field.v_xy(self.influence.scale_pt(pt));

        if let Some(child) = self.child_at_point(data, pt) {
            let child = &data[*child];
            let child_area = child.influence.area();

            let child_weight = if this_area.is_finite() && this_area > 0. {
                let closeness = child.influence.edge_closeness(pt);
                0.05 * closeness * child_area / this_area
            } else {
                1.
            };

            let child_vector = child_weight * child.field_vector(data, pt);

            vector += child_vector;
        }

        vector
    }

    pub fn child_at_point<D>(&self, data: &D, pt: Point2<f32>) -> Option<&FieldID>
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
    pub fn new(min: f32, max: f32) -> Self {
        RangeOF {
            min: min.into(),
            max: max.into(),
            phantom: std::marker::PhantomData,
        }
    }
    pub fn contains(&self, value: f32) -> bool {
        value >= *self.min && value <= *self.max
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
    pub fn edge_closeness(&self, pt: Point2<f32>) -> f32 {
        let c = self.center();
        let dc = c - pt;
        let pc_width = 1. - 2. * dc.x.abs() / self.width();
        let pc_height = 1. - 2. * dc.y.abs() / self.height();

        pc_width.min(pc_height)
    }
    pub fn quadrants(&self) -> [Self; 4] {
        let center = self.center();
        let x0 = *self.min.x;
        let y0 = *self.min.y;
        let x1 = *self.max.x;
        let y1 = *self.max.y;
        [
            AreaOF {
                min: Point2::new(x0.into(), y0.into()),
                max: Point2::new(center.x.into(), center.y.into()),
            },
            AreaOF {
                min: Point2::new(center.x.into(), y0.into()),
                max: Point2::new(x1.into(), center.y.into()),
            },
            AreaOF {
                min: Point2::new(x0.into(), center.y.into()),
                max: Point2::new(center.x.into(), y1.into()),
            },
            AreaOF {
                min: Point2::new(center.x.into(), center.y.into()),
                max: Point2::new(x1.into(), y1.into()),
            },
        ]
    }
    pub fn center(&self) -> Point2<f32> {
        let xc = *self.min.x + (*self.max.x - *self.min.x) / 2.;
        let yc = *self.min.y + (*self.max.y - *self.min.y) / 2.;
        let x = xc.is_finite().then_some(xc).unwrap_or_default();
        let y = yc.is_finite().then_some(yc).unwrap_or_default();

        Point2::new(x, y)
    }
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
    pub fn un_scale_pt(&self, pt: Point2<f32>) -> Point2<f32> {
        let dx = self.width();
        let dy = self.height();

        let x = pt.x * dx;
        let y = pt.y * dy;
        Point2::new(x, y)
    }
    pub fn scale_pt(&self, pt: Point2<f32>) -> Point2<f32> {
        let dx = self.width();
        let dy = self.height();

        let x = pt.x / dx;
        let y = pt.y / dy;
        Point2::new(x, y)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn width(&self) -> f32 {
        *(self.max.x - self.min.x)
    }

    pub fn height(&self) -> f32 {
        *(self.max.y - self.min.y)
    }
    pub fn contains(&self, pt: Point2<f32>) -> bool {
        pt.x >= *self.min.x && pt.x <= *self.max.x && pt.y >= *self.min.y && pt.y <= *self.max.y
    }

    pub fn xy_ranges(&self) -> (XRangeOF, YRangeOF) {
        (self.x_range(), self.y_range())
    }
    pub fn y_range(&self) -> YRangeOF {
        RangeOF::new(*self.min.y, *self.max.y)
    }
    pub fn x_range(&self) -> XRangeOF {
        RangeOF::new(*self.min.x, *self.max.x)
    }
}
