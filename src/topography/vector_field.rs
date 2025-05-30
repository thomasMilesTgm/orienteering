use std::ops::Deref;

use enum_dispatch::enum_dispatch;
use nalgebra::{Point2, Vector2};
use rand::Rng;

use crate::{proc_gen::ProceduralValue, utils::proc_field};

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
    fn v_xy(&self, xy: Point2<f32>) -> Vector2<f32>;
}

impl<T: Deref<Target = G>, G: Field> Field for T {
    fn v_xy(&self, xy: Point2<f32>) -> Vector2<f32> {
        self.deref().v_xy(xy)
    }
}

impl<Vx: VectorFn, Vy: VectorFn> Field for GeneralVectorField<Vx, Vy> {
    fn v_xy(&self, xy: Point2<f32>) -> Vector2<f32> {
        Vector2::new(self.vx.v_xy(xy), self.vy.v_xy(xy))
    }
}

impl<Vx: VectorFn, Vy: VectorFn> ProceduralValue for GeneralVectorField<Vx, Vy> {
    fn from_rng<T: Rng>(rng: &mut T) -> Self {
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
impl VectorField {
    pub fn new<T: Into<FieldType>>(field: T) -> Self {
        VectorField {
            field: field.into(),
        }
    }
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
    Stationary(StationaryField),
}

#[derive(Debug, Clone, derive_more::Deref)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HillyBowlField {
    field: GeneralVectorField<ConstantDir, SinX2Y2>,
}

proc_field!(HillyBowlField);

#[derive(Debug, Clone, derive_more::Deref)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SaddleField {
    field: GeneralVectorField<Kx, Ky>,
}
proc_field!(SaddleField);

#[derive(Debug, Clone, derive_more::Deref)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceField {
    field: GeneralVectorField<Kx, Ky>,
}

proc_field!(SourceField);

#[derive(Debug, Clone, derive_more::Deref)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SinkField {
    field: GeneralVectorField<Kx, Ky>,
}

proc_field!(SinkField);

/// A field that rotates ccw about the origin.
#[derive(Debug, Clone, derive_more::Deref)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CircularField {
    field: GeneralVectorField<Ky, Kx>,
}

impl ProceduralValue for CircularField {
    fn from_rng<T: Rng>(rng: &mut T) -> Self {
        let mut vx = Ky::from_rng(rng);
        vx.make_negative();
        let mut vy = Kx::from_rng(rng);
        vy.make_positive();
        Self {
            field: GeneralVectorField { vx, vy },
        }
    }
}

/// A field that is zero everywhere.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StationaryField;

impl ProceduralValue for StationaryField {
    fn from_rng<T: Rng>(_: &mut T) -> Self {
        // TODO: Add quantum fluctuations within stationary fields to randomly disturb gubbins.
        Self
    }
}

impl Field for StationaryField {
    fn v_xy(&self, _: Point2<f32>) -> Vector2<f32> {
        Vector2::zeros()
    }
}
