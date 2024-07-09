/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::atanf::eatanf;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::atan2f::vatan2q_f32;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};

fn do_atan2f(y: f32, x: f32) -> f32 {
    if x == 0. {
        if y > 0. {
            return std::f32::consts::FRAC_PI_2;
        }
        if y < 0. {
            return -std::f32::consts::FRAC_PI_2;
        }
        if y == 0. {
            return 0f32;
        }
    }
    let rad = eatanf(y / x);
    return if x > 0f32 {
        rad
    } else if x < 0f32 && y >= 0f32 {
        std::f32::consts::PI + rad
    } else {
        // if x < 0. && y < 0.
        -std::f32::consts::PI + rad
    };
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
fn do_atan2f_neon(y: f32, x: f32) -> f32 {
    unsafe {
        let vy = vdupq_n_f32(y);
        let vx = vdupq_n_f32(x);
        vgetq_lane_f32::<0>(vatan2q_f32(vy, vx))
    }
}

/// Computes atan2 between vector, *ULP 1.0*
#[inline]
pub fn eatan2f(y: f32, x: f32) -> f32 {
    let mut _dispatcher: fn(f32, f32) -> f32 = do_atan2f;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_atan2f_neon;
    }
    _dispatcher(y, x)
}
