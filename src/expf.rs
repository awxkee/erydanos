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
use crate::_mm_exp_ps;
use crate::generalf::{mlaf, pow2if, rintfk};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::vexpq_f32;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

pub(crate) const EXP_POLY_1_S: f32 = 2f32;
pub(crate) const EXP_POLY_2_S: f32 = 0.16666707f32;
pub(crate) const EXP_POLY_3_S: f32 = -0.002775669f32;
pub(crate) const EXP_POLY_4_S: f32 = 6.6094115e-5f32;
pub(crate) const EXP_POLY_5_S: f32 = 1.6546869e-6f32;

pub const L2U_F: f32 = 0.693_145_751_953_125;
pub const L2L_F: f32 = 1.428_606_765_330_187_045_e-6;
pub const R_LN2_F: f32 = std::f32::consts::LOG2_E;

#[inline]
fn do_exp(d: f32) -> f32 {
    let qf = rintfk(d * R_LN2_F);
    let q = qf as i32;
    let r = mlaf(qf, -L2U_F, d);
    let r = mlaf(qf, -L2L_F, r);

    let f = r * r;
    // Poly for u = r*(exp(r)+1)/(exp(r)-1)
    let mut u = EXP_POLY_5_S;
    u = mlaf(u, f, EXP_POLY_4_S);
    u = mlaf(u, f, EXP_POLY_3_S);
    u = mlaf(u, f, EXP_POLY_2_S);
    u = mlaf(u, f, EXP_POLY_1_S);
    let u = 1f32 + 2f32 * r / (u - r);
    let i2 = pow2if(q);
    let mut r = u * i2;
    if d < -87f32 {
        r = 0f32;
    }
    if d > 88f32 {
        r = f32::INFINITY;
    }
    r
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn do_exp_neon(d: f32) -> f32 {
    unsafe {
        let ld = vdupq_n_f32(d);
        vgetq_lane_f32::<0>(vexpq_f32(ld))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_exp_sse(d: f32) -> f32 {
    unsafe {
        let v = _mm_set1_ps(d);
        let value = _mm_exp_ps(v);
        let ex = f32::from_bits(_mm_extract_ps::<0>(value) as u32);
        ex
    }
}

/// Computes exp for an argument *ULP 1.0*
#[inline]
pub fn eexpf(d: f32) -> f32 {
    let mut _dispatcher: fn(f32) -> f32 = do_exp;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = do_exp_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_exp_sse;
    }
    _dispatcher(d)
}
