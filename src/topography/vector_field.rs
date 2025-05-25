use nalgebra::{Point2, Vector2};
use rand::Rng;

use super::vector_function::*;

/// A 2D Vector Field with arbitrary x and y component functions.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GeneralVectorField<Vx: VectorFn, Vy: VectorFn> {
    pub vx: Vx,
    pub vy: Vy,
}

impl<Vx: VectorFn, Vy: VectorFn> GeneralVectorField<Vx, Vy> {
    pub fn v_xy(&self, xy: Point2<f32>) -> Vector2<f32> {
        Vector2::new(self.vx.v_xy(xy), self.vy.v_xy(xy))
    }
    pub fn from_rng<T: Rng>(rng: &mut T) -> Self {
        let vx = Vx::from_rng(rng);
        let vy = Vy::from_rng(rng);
        Self { vx, vy }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorField {
    field: FieldType,
}

/// Vector field types used to generate terrain
#[derive(Debug, Clone, derive_more::From)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FieldType {
    HillyBowl(HillyBowlField),
    Saddle(SaddleField),
    Source(SourceField),
    Sink(SinkField),
    Circular(CircularField),
}

#[derive(Debug, Clone, derive_more::Deref)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HillyBowlField {
    field: GeneralVectorField<ConstantDir, SinX2Y2>,
}

#[derive(Debug, Clone, derive_more::Deref)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SaddleField {
    field: GeneralVectorField<Kx, Ky>,
}

#[derive(Debug, Clone, derive_more::Deref)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceField {
    field: GeneralVectorField<Kx, Ky>,
}

#[derive(Debug, Clone, derive_more::Deref)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SinkField {
    field: GeneralVectorField<Kx, Ky>,
}

#[derive(Debug, Clone, derive_more::Deref)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CircularField {
    field: GeneralVectorField<Kx, Ky>,
}
