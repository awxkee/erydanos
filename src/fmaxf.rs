/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

pub fn efmaxf(x: f32, y: f32) -> f32 {
    if y.is_nan() || (x > y) {
        x
    } else {
        y
    }
}