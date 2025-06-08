//! # Visualize scalar fields in Bevy.
//!
//! The scalar field in orienteering can be thought of as the gravitational potential of the world
//! map, given this we can use it's gradient vector field to generate contours on the world map.

use bevy::prelude::*;
use orienteering_bevy::camera::CameraPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(CameraPlugin)
        //..
        .run();
}
