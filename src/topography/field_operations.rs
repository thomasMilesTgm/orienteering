//! Operations for working with vector fields

use crate::topography::Field;
use nalgebra::{Point2, Vector2};

const DT: f32 = 0.1;
const CLOSE: f32 = 1.;
const MAX_LENGTH: f32 = 10000.;

#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineIntegral {
    pub points: Vec<FieldPoint>,
}

#[derive(Debug, Clone)]
pub struct LineIntegralCfg {
    /// Starting point
    pub p0: Point2<f32>,
    /// Time step for the integration
    pub dt: f32,
    /// Maximum length of the line integral
    pub max_length: f32,
    /// If the line gets within this distance, stop solving
    pub stop_within: f32,
}

impl LineIntegralCfg {
    pub fn new(p0: Point2<f32>) -> Self {
        Self {
            p0,
            dt: DT,
            stop_within: CLOSE,
            max_length: MAX_LENGTH,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldPoint {
    pub xy: Point2<f32>,
    pub v_xy: Vector2<f32>,
    pub divergence: f32,
    pub curl: f32,
}

impl LineIntegral {
    pub fn solve<F: Field>(field: &F, cfg: &LineIntegralCfg) -> Self {
        let mut new = Self::default();
        let mut left_start = false;
        let mut xy0 = cfg.p0;
        let mut length = cfg.max_length;

        while length > 0. {
            if !left_start {
                left_start = new
                    .points
                    .iter()
                    .any(|pt| (xy0 - pt.xy).magnitude() > 2. * CLOSE);
            } else if (xy0 - new.points[0].xy).magnitude() < CLOSE {
                println!("Contour closed at {:?}", xy0);
                break;
            }

            let f_xy = field.v_xy(xy0) * DT;

            if f_xy.magnitude() < 0.01 {
                println!("Contour became stationary at {:?}", xy0);
                break;
            }

            let xy = xy0 + f_xy; // Next Point
            let grad = xy - xy0; // Gradient, ∇
            let divergence = grad.dot(&f_xy); // ∇ • F
            let curl = grad.x * f_xy.y - f_xy.x * grad.y; // |∇ × F|

            let field_pt = FieldPoint {
                xy,
                v_xy: f_xy,
                divergence,
                curl,
            };

            new.points.push(field_pt);

            xy0 = xy;
            length -= f_xy.magnitude();
        }
        new
    }
}
