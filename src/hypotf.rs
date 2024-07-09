/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::abs::eabsf;
use crate::fmaxf::efmaxf;
use crate::fminf::efminf;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::hypotf::vhypotq_f32;
use crate::sqrtf::esqrtf;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};

#[inline]
fn do_hypotf(x: f32, y: f32) -> f32 {
    let x = eabsf(x);
    let y = eabsf(y);
    let max = efmaxf(x, y);
    let min = efminf(x, y);
    let r = min / max;
    let ret = max * esqrtf(1f32 + r * r);

    if (x == f32::INFINITY) || (y == f32::INFINITY) {
        f32::INFINITY
    } else if x.is_nan() || y.is_nan() {
        f32::NAN
    } else if min == 0. {
        max
    } else if ret.is_nan() {
        f32::INFINITY
    } else {
        ret
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline]
fn do_hypotf_neon(x: f32, y: f32) -> f32 {
    unsafe {
        let vx = vdupq_n_f32(x);
        let vy = vdupq_n_f32(y);
        vgetq_lane_f32::<0>(vhypotq_f32(vx, vy))
    }
}

/// Computes 2D Euclidian Distance *ULP 0.5*
#[inline]
pub fn ehypotf(x: f32, y: f32) -> f32 {
    let mut _dispatcher: fn(f32, f32) -> f32 = do_hypotf;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_hypotf_neon;
    }
    _dispatcher(x, y)
}
