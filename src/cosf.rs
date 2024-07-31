/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::_mm_cos_ps;
use crate::generalf::{mlaf, rintfk, IsNegZero};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vcosq_f32;
use crate::sinf::{SIN_POLY_1_S, SIN_POLY_2_S, SIN_POLY_3_S, SIN_POLY_4_S, SIN_POLY_5_S};
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

pub(crate) const PI_A_F: f32 = 3.140_625;
pub(crate) const PI_B_F: f32 = 0.000_967_025_756_835_937_5;
pub(crate) const PI_C_F: f32 = 6.277_114_152_908_325_195_3_e-7;
pub(crate) const PI_D_F: f32 = 1.215_420_125_655_342_076_2_e-10;

#[inline]
fn do_cos(d: f32) -> f32 {
    let q = 1 + 2 * rintfk(std::f32::consts::FRAC_1_PI * d - 0.5) as i32;
    let qf = q as f32;
    let mut r = mlaf(qf, -PI_A_F * 0.5, d);
    r = mlaf(qf, -PI_B_F * 0.5, r);
    r = mlaf(qf, -PI_C_F * 0.5, r);
    r = mlaf(qf, -PI_D_F * 0.5, r);

    let x2 = r * r;

    if q & 2 == 0 {
        r = -r;
    }

    let mut u = SIN_POLY_5_S;
    u = mlaf(u, x2, SIN_POLY_4_S);
    u = mlaf(u, x2, SIN_POLY_3_S);
    u = mlaf(u, x2, SIN_POLY_2_S);
    u = mlaf(u, x2, SIN_POLY_1_S);
    u = mlaf(u, x2 * r, r);
    if u.isnegzero() {
        return 0.;
    }
    u
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline]
fn do_cos_neon(d: f32) -> f32 {
    unsafe {
        let j = vdupq_n_f32(d);
        vgetq_lane_f32::<0>(vcosq_f32(j))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_cos_sse(d: f32) -> f32 {
    unsafe {
        let v = _mm_set1_ps(d);
        let value = _mm_cos_ps(v);
        let ex = f32::from_bits(_mm_extract_ps::<0>(value) as u32);
        ex
    }
}

/// Cosine function
///
/// The error bound of the returned value is `1.5 ULP`.
pub fn ecosf(d: f32) -> f32 {
    let mut _dispatcher: fn(f32) -> f32 = do_cos;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_cos_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_cos_sse;
    }
    _dispatcher(d)
}
