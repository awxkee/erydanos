/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::abs::eabsf;
use crate::asinf::easinf;

#[inline]
fn do_acosf(x: f32) -> f32 {
    if x > 0f32 {
        std::f32::consts::FRAC_PI_2 - easinf(x)
    } else {
        let v = eabsf(x);
        std::f32::consts::FRAC_PI_2 + easinf(v)
    }
}

/// Computes acos for an argument, *ULP 30*
#[inline]
pub fn eacosf(d: f32) -> f32 {
    let mut _dispatcher: fn(f32) -> f32 = do_acosf;
    _dispatcher(d)
}
