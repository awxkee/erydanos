/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::abs::eabsf;
use crate::floor::F1_23;
use crate::generalf::copysignfk;

#[inline]
pub fn eceilf(x: f32) -> f32 {
    let mut fr = x - (x as i32 as f32);
    fr = if fr <= 0. { fr } else { fr - 1. };
    if x.is_infinite() || (eabsf(x) >= F1_23) {
        x
    } else {
        copysignfk(x - fr, x)
    }
}
