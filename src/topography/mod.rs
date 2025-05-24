//! # Topography
//!
//! Defines the topography of the world, i.e. the shape of terrain, along with biomes, structures,
//! features, etc.

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WorldMap {}

pub mod vector_field {
    use nalgebra::{Isometry2, Point2, Vector2};

    use super::vector_function::*;

    pub struct HillyBowlField {
        pub field: VectorField<ConstantDir, SinX2Y2>,
    }

    /// A 2D Vector Field with arbitrary x and y component functions.
    pub struct VectorField<Vx: VectorFn, Vy: VectorFn> {
        pub vx: Vx,
        pub vy: Vy,
        pub tf: Isometry2<f32>,
    }

    impl<Vx: VectorFn, Vy: VectorFn> VectorField<Vx, Vy> {
        pub fn v_xy(&self, xy: Point2<f32>) -> Vector2<f32> {
            let xy_local = self.tf.transform_point(&xy);
            Vector2::new(self.vx.v_xy(xy_local), self.vy.v_xy(xy_local))
        }
    }
}

pub mod vector_function {
    use nalgebra::Point2;
    use rand::SeedableRng;

    use crate::seed::MapSeed;

    /// A vector field function, used to define the x or y component of a vector field at a given
    /// point in 2D space.
    pub trait VectorFn: SeedableRng<Seed = MapSeed> {
        fn v_xy(&self, xy: Point2<f32>) -> f32;
    }

    pub struct ConstantDir {
        pub k: f32,
    }

    impl VectorFn for ConstantDir {
        fn v_xy(&self, _xy: Point2<f32>) -> f32 {
            self.k
        }
    }

    impl SeedableRng for ConstantDir {
        type Seed = MapSeed;
        fn from_seed(seed: Self::Seed) -> Self {
            todo!()
        }
    }

    /// V = k * sin(a * x^2 + b * y^2)
    pub struct SinX2Y2 {
        pub k: f32,
        pub a: f32,
        pub b: f32,
    }

    impl VectorFn for SinX2Y2 {
        fn v_xy(&self, xy: Point2<f32>) -> f32 {
            self.k * (self.a * xy.x.powi(2) + self.b * xy.y.powi(2)).sin()
        }
    }

    impl SeedableRng for SinX2Y2 {
        type Seed = MapSeed;
        fn from_seed(seed: Self::Seed) -> Self {
            todo!()
        }
    }

    /// V = j * x + k * y
    pub struct JxKy {
        pub j: f32,
        pub k: f32,
    }

    impl VectorFn for JxKy {
        fn v_xy(&self, xy: Point2<f32>) -> f32 {
            self.j * xy.x + self.k * xy.y
        }
    }
    impl SeedableRng for JxKy {
        type Seed = MapSeed;
        fn from_seed(seed: Self::Seed) -> Self {
            todo!()
        }
    }
}
