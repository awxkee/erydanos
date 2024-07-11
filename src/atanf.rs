/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::generalf::mlaf;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vatanq_f32;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};

pub const ATAN_POLY_1_F: f32 = 0.999999871164f32;
pub const ATAN_POLY_2_F: f32 = -0.333325240026f32;
pub const ATAN_POLY_3_F: f32 = 0.199848846856f32;
pub const ATAN_POLY_4_F: f32 = -0.141548060419f32;
pub const ATAN_POLY_5_F: f32 = 0.104775391987f32;
pub const ATAN_POLY_6_F: f32 = -0.0719438454246f32;
pub const ATAN_POLY_7_F: f32 = 0.0393454131479f32;
pub const ATAN_POLY_8_F: f32 = -0.0141523480362f32;
pub const ATAN_POLY_9_F: f32 = 0.00239813901251f32;

#[inline]
fn do_atanf(d: f32) -> f32 {
    let mut x = d;
    let q = if x < 0f32 {
        x = -x;
        1
    } else {
        0
    };
    let c = x;
    if x > 1f32 {
        x = 1f32 / x;
    }
    let x2 = x * x;
    let mut u = ATAN_POLY_9_F;
    u = mlaf(u, x2, ATAN_POLY_8_F);
    u = mlaf(u, x2, ATAN_POLY_7_F);
    u = mlaf(u, x2, ATAN_POLY_6_F);
    u = mlaf(u, x2, ATAN_POLY_5_F);
    u = mlaf(u, x2, ATAN_POLY_4_F);
    u = mlaf(u, x2, ATAN_POLY_3_F);
    u = mlaf(u, x2, ATAN_POLY_2_F);
    u = mlaf(u, x2, ATAN_POLY_1_F);
    u = u * x;
    u = if c > 1f32 {
        std::f32::consts::FRAC_PI_2 - u
    } else {
        u
    };
    if q & 1 != 0 {
        u = -u;
    }
    u
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
fn do_atanf_neon(y: f32) -> f32 {
    unsafe {
        let vy = vdupq_n_f32(y);
        vgetq_lane_f32::<0>(vatanq_f32(vy))
    }
}

/// Computes Atan function with *ULP 2.0* error
#[inline]
pub fn eatanf(d: f32) -> f32 {
    let mut _dispatcher: fn(f32) -> f32 = do_atanf;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_atanf_neon;
    }
    _dispatcher(d)
}
