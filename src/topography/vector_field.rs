use std::ops::Deref;

use enum_dispatch::enum_dispatch;
use nalgebra::{Point2, Vector2};
use rand::Rng;

use crate::utils::of32;

use super::vector_function::*;

/// A 2D Vector Field with arbitrary x and y component functions.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GeneralVectorField<Vx: VectorFn, Vy: VectorFn> {
    pub vx: Vx,
    pub vy: Vy,
}

#[enum_dispatch]
pub trait Field {
    fn v_xy(&self, xy: Point2<of32>) -> Vector2<of32>;
}

impl<T: Deref<Target = G>, G: Field> Field for T {
    fn v_xy(&self, xy: Point2<of32>) -> Vector2<of32> {
        self.deref().v_xy(xy)
    }
}

impl<Vx: VectorFn, Vy: VectorFn> Field for GeneralVectorField<Vx, Vy> {
    fn v_xy(&self, xy: Point2<of32>) -> Vector2<of32> {
        Vector2::new(self.vx.v_xy(xy), self.vy.v_xy(xy))
    }
}

impl<Vx: VectorFn, Vy: VectorFn> GeneralVectorField<Vx, Vy> {
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

impl std::ops::Deref for VectorField {
    type Target = FieldType;

    fn deref(&self) -> &Self::Target {
        &self.field
    }
}

/// Vector field types used to generate terrain
#[derive(Debug, Clone)]
#[enum_dispatch(Field)]
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
