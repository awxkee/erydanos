/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::_mm_sin_ps;
use crate::cosf::{PI_A_F, PI_B_F, PI_C_F, PI_D_F};
use crate::generalf::{mlaf, rintfk, IsNegZero};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::vsinq_f32;
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

pub const SIN_POLY_1_S: f32 = -0.16666667f32;
pub const SIN_POLY_2_S: f32 = 0.0083333375f32;
pub const SIN_POLY_3_S: f32 = -0.00019841341f32;
pub const SIN_POLY_4_S: f32 = 2.7551241e-6f32;
pub const SIN_POLY_5_S: f32 = -2.4535176e-8f32;

#[inline]
fn do_sin(d: f32) -> f32 {
    let qf = rintfk(std::f32::consts::FRAC_1_PI * d);
    let q = qf as i32;
    let mut r = mlaf(qf, -PI_A_F, d);
    r = mlaf(qf, -PI_B_F, r);
    r = mlaf(qf, -PI_C_F, r);
    r = mlaf(qf, -PI_D_F, r);

    let x2 = r * r;

    if (q & 1) != 0 {
        r = -r;
    }

    let mut u = SIN_POLY_5_S;
    u = mlaf(u, x2, SIN_POLY_4_S);
    u = mlaf(u, x2, SIN_POLY_3_S);
    u = mlaf(u, x2, SIN_POLY_2_S);
    u = mlaf(u, x2, SIN_POLY_1_S);
    u = mlaf(u, x2 * r, r);
    if u.isnegzero() {
        return 0f32;
    }
    u
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn do_sin_neon(d: f32) -> f32 {
    unsafe {
        let j = vdupq_n_f32(d);
        vgetq_lane_f32::<0>(vsinq_f32(j))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_sin_sse(d: f32) -> f32 {
    unsafe {
        let v = _mm_set1_ps(d);
        let value = _mm_sin_ps(v);
        let ex = f32::from_bits(_mm_extract_ps::<0>(value) as u32);
        ex
    }
}

/// Computes sine function with error bound *ULP 1.2*
#[inline]
pub fn esinf(d: f32) -> f32 {
    let mut _dispatcher: fn(f32) -> f32 = do_sin;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = do_sin_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_sin_sse;
    }
    _dispatcher(d)
}
