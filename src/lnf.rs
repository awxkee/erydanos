/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::_mm_ln_ps;
use crate::generalf::{ilogb2kf, ldexp3kf, mlaf};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::vlnq_f32;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

pub const LN_POLY_1_F: f32 = 2f32;
pub const LN_POLY_2_F: f32 = 0.6666677f32;
pub const LN_POLY_3_F: f32 = 0.40017125f32;
pub const LN_POLY_4_F: f32 = 0.28523374f32;
pub const LN_POLY_5_F: f32 = 0.23616748f32;

#[inline]
fn do_ln(d: f32) -> f32 {
    // ln(𝑥)=ln(𝑎)+𝑛ln(2)
    let n = ilogb2kf(d * (1. / 0.75));
    let a = ldexp3kf(d, -n);

    let x = (a - 1.) / (a + 1.);
    let x2 = x * x;
    let mut u = LN_POLY_5_F;
    u = mlaf(u, x2, LN_POLY_4_F);
    u = mlaf(u, x2, LN_POLY_3_F);
    u = mlaf(u, x2, LN_POLY_2_F);
    u = mlaf(u, x2, LN_POLY_1_F);
    return if d == 0f32 {
        f32::NEG_INFINITY
    } else if (d < 0.) || d.is_nan() {
        f32::NAN
    } else if d.is_infinite() {
        f32::INFINITY
    } else {
        x * u + std::f32::consts::LN_2 * (n as f32)
    };
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn do_lnf_neon(x: f32) -> f32 {
    unsafe {
        let vx = vdupq_n_f32(x);
        vgetq_lane_f32::<0>(vlnq_f32(vx))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_ln_sse(d: f32) -> f32 {
    unsafe {
        let v = _mm_set1_ps(d);
        let value = _mm_ln_ps(v);
        let ex = f32::from_bits(_mm_extract_ps::<0>(value) as u32);
        ex
    }
}

/// Computes natural logarithm for an argument *ULP 1.0*
pub fn elnf(d: f32) -> f32 {
    let mut _dispatcher: fn(f32) -> f32 = do_ln;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = do_lnf_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_ln_sse;
    }
    _dispatcher(d)
}
