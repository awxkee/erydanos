/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::abs::eabs;
use crate::floor::{D1_31, D1_52};
use crate::generalf::copysignk;

#[inline]
pub fn eceil(x: f64) -> f64 {
    let mut fr = x - D1_31 * ((x * (1. / D1_31)) as i32 as f64);
    fr -= fr as i32 as f64;
    fr = if fr <= 0. { fr } else { fr - 1. };
    if x.is_infinite() || (eabs(x) >= D1_52) {
        x
    } else {
        copysignk(x - fr, x)
    }
}
