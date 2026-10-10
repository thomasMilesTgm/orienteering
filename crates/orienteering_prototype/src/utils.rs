//! Various utilities.

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
