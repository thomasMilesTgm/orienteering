//! Various utilities.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dir {
    X,
    Y,
}

impl Dir {
    pub fn r(&self, pt: &Point2<f32>) -> f32 {
        match self {
            Dir::X => pt.x,
            Dir::Y => pt.y,
        }
    }
    pub fn to_pt(&self, value: f32) -> Point2<f32> {
        match self {
            Dir::X => Point2::new(value, 0.),
            Dir::Y => Point2::new(0., value),
        }
    }
    pub fn to_vector(&self, value: f32) -> Vector2<f32> {
        match self {
            Dir::X => Vector2::new(value, 0.),
            Dir::Y => Vector2::new(0., value),
        }
    }
}

#[allow(non_camel_case_types)]
pub type of32 = ordered_float::OrderedFloat<f32>;

/// Implements [`ProceuralValue`](crate::proc_gen::ProceduralValue) for a type that contains a
/// `GeneralVectorField` called `field`.
macro_rules! proc_field {
    ($ty:ident) => {
        impl ProceduralValue for $ty {
            fn from_rng<T: Rng>(rng: &mut T) -> Self {
                Self {
                    field: GeneralVectorField::from_rng(rng),
                }
            }
        }
    };
}

pub(crate) use proc_field;
