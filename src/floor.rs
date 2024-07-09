/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::abs::{eabs, eabsf};
use crate::generalf::{copysignfk, copysignk};

pub(crate) const F1_23: f32 = (1u32 << 23) as f32;
pub(crate) const D1_31: f64 = (1u64 << 31) as f64;
pub(crate) const D1_52: f64 = (1u64 << 52) as f64;

/// Round to integer towards minus infinity
#[inline]
pub fn efloorf(x: f32) -> f32 {
    let mut fr = x - (x as i32 as f32);
    fr = if fr < 0. { fr + 1. } else { fr };
    if x.is_infinite() || (eabsf(x) >= F1_23) {
        x
    } else {
        copysignfk(x - fr, x)
    }
}

/// Round to integer towards minus infinity
#[inline]
pub fn efloor(x: f64) -> f64 {
    let mut fr = x - D1_31 * ((x * (1. / D1_31)) as i32 as f64);
    fr -= fr as i32 as f64;
    fr = if fr < 0. { fr + 1. } else { fr };
    if x.is_infinite() || (eabs(x) >= D1_52) {
        x
    } else {
        copysignk(x - fr, x)
    }
}
