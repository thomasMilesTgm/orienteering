//! Seed editing, the steps slider and keys, panning and zooming.

use super::*;

pub(crate) fn edit_seed(
    mut controls: ResMut<Controls>,
    mut keys: MessageReader<KeyboardInput>,
    mouse: Res<ButtonInput<MouseButton>>,
    seed_box: Single<&RelativeCursorPosition, With<SeedBox>>,
) {
    if mouse.just_pressed(MouseButton::Left) {
        controls.draft = if seed_box.cursor_over {
            Some(controls.seed.clone())
        } else {
            None
        };
    }
    for key in keys.read() {
        if key.state != ButtonState::Pressed {
            continue;
        }
        let Some(draft) = controls.draft.as_mut() else {
            continue;
        };
        match &key.logical_key {
            Key::Character(c) if !c.chars().any(char::is_control) => draft.push_str(c),
            Key::Space => draft.push(' '),
            Key::Backspace => {
                draft.pop();
            }
            Key::Enter => {
                let seed = draft.trim().to_string();
                if !seed.is_empty() {
                    controls.seed = seed;
                }
                controls.draft = None;
            }
            Key::Escape => controls.draft = None,
            _ => {}
        }
    }
}

pub(crate) fn drag_slider(
    mut controls: ResMut<Controls>,
    mouse: Res<ButtonInput<MouseButton>>,
    track: Single<&RelativeCursorPosition, With<Track>>,
    time: Res<Time>,
) {
    if mouse.just_pressed(MouseButton::Left) && track.cursor_over {
        controls.dragging_slider = true;
        controls.playing = false;
    }
    if !mouse.pressed(MouseButton::Left) {
        controls.dragging_slider = false;
    }
    if controls.dragging_slider
        && let Some(n) = track.normalized
    {
        controls.stage = ((n.x as f64 + 0.5) * steps() as f64).clamp(0., steps() as f64);
        controls.last_interaction = time.elapsed_secs();
    }
}

pub(crate) fn step_keys(
    mut controls: ResMut<Controls>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut orbit: ResMut<Orbit>,
) {
    let last = steps() as f64;
    if controls.draft.is_none() {
        if keys.just_pressed(KeyCode::ArrowLeft) {
            controls.stage = (controls.stage.ceil() - 1.).max(0.);
            controls.playing = false;
        }
        if keys.just_pressed(KeyCode::ArrowRight) {
            controls.stage = (controls.stage.floor() + 1.).min(last);
            controls.playing = false;
        }
        if keys.just_pressed(KeyCode::KeyG) {
            controls.gpu = !controls.gpu;
        }
        if keys.just_pressed(KeyCode::KeyT) {
            controls.plates = !controls.plates;
        }
        if keys.just_pressed(KeyCode::Digit3) {
            controls.three_d = !controls.three_d;
        }
        if keys.just_pressed(KeyCode::KeyE) {
            orbit.exaggeration += 1;
        }
        if keys.just_pressed(KeyCode::Space) {
            controls.playing = !controls.playing;
            if controls.playing && controls.stage >= last {
                controls.stage = 0.;
            }
        }
    }
    if controls.playing {
        controls.stage = (controls.stage + PLAY_SPEED * time.delta_secs_f64()).min(last);
        controls.playing = controls.stage < last;
        controls.last_interaction = time.elapsed_secs();
    }
}

pub(crate) fn pan_and_zoom(
    mut controls: ResMut<Controls>,
    mut view: ResMut<View>,
    mut wheel: MessageReader<MouseWheel>,
    mouse: Res<ButtonInput<MouseButton>>,
    window: Single<&Window, With<PrimaryWindow>>,
    panels: Query<&RelativeCursorPosition, With<Panel>>,
    time: Res<Time>,
) {
    if controls.three_d {
        wheel.clear();
        return;
    }
    let Some(cursor) = window.cursor_position() else {
        wheel.clear();
        return;
    };
    let over_ui = panels.iter().any(|r| r.cursor_over);

    let mut zoom = 0.;
    for w in wheel.read() {
        zoom += match w.unit {
            MouseScrollUnit::Line => w.y as f64,
            MouseScrollUnit::Pixel => w.y as f64 / 40.,
        };
    }
    if zoom != 0. && !over_ui {
        // Keep the point under the cursor fixed.
        let anchor = view.world(&window, cursor);
        let scale = (view.m_per_px * 1.2f64.powf(-zoom)).clamp(SCALE_RANGE.0, SCALE_RANGE.1);
        view.center = anchor + (view.center - anchor) * (scale / view.m_per_px);
        view.m_per_px = scale;
        controls.last_interaction = time.elapsed_secs();
    }

    if mouse.just_pressed(MouseButton::Left) && !over_ui && !controls.dragging_slider {
        controls.panning = Some(cursor);
    }
    if !mouse.pressed(MouseButton::Left) {
        controls.panning = None;
    }
    if let Some(last) = controls.panning {
        let d = cursor - last;
        if d != Vec2::ZERO {
            let scale = view.m_per_px;
            view.center += Vector2::new(-d.x as f64, d.y as f64) * scale;
            controls.panning = Some(cursor);
            controls.last_interaction = time.elapsed_secs();
        }
    }
}
