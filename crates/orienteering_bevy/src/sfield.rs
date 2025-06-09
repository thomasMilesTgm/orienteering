//! Scalar field plugin

use bevy::prelude::*;

pub struct ScalarFieldPlugin;

impl Plugin for ScalarFieldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, generate_map)
    }
}

/// Gravitational potential field for the world map
struct GPEField {}

fn generate_map(commands: &mut Commands) {}
