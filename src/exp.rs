/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::generalf::{mlaf, pow2i, rintk};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vexpq_f64;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::{_mm_exp_pd, _mm_extract_pd};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::{vdupq_n_f64, vgetq_lane_f64};
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

pub(crate) const EXP_POLY_1_D: f64 = 2f64;
pub(crate) const EXP_POLY_2_D: f64 = 0.16666666666666674f64;
pub(crate) const EXP_POLY_3_D: f64 = -0.0027777777777777614f64;
pub(crate) const EXP_POLY_4_D: f64 = 6.613756613755705e-5f64;
pub(crate) const EXP_POLY_5_D: f64 = -1.6534391534392554e-6f64;
pub(crate) const EXP_POLY_6_D: f64 = 4.17535139757361979584e-8f64;
pub(crate) const EXP_POLY_7_D: f64 = -1.05683802773749863697e-9f64;
pub(crate) const EXP_POLY_8_D: f64 = 2.67650730613693576657e-11f64;
pub(crate) const EXP_POLY_9_D: f64 = 1.71721241125556891283e-14;
pub(crate) const EXP_POLY_10_D: f64 = -6.77936059264516573366e-13f64;

pub(crate) const L2_U: f64 = 0.693_147_180_559_662_956_511_601_805_686_950_683_593_75;
pub(crate) const L2_L: f64 = 0.282_352_905_630_315_771_225_884_481_750_134_360_255_254_120_68_e-12;
pub(crate) const R_LN2: f64 =
    1.442_695_040_888_963_407_359_924_681_001_892_137_426_645_954_152_985_934_135_449_406_931;

#[inline]
fn do_exp(d: f64) -> f64 {
    let qf = rintk(d * R_LN2);
    let q = qf as i32;

    let mut r = mlaf(qf, -L2_U, d);
    r = mlaf(qf, -L2_L, r);

    let f = r * r;
    // Poly for u = r*(exp(r)+1)/(exp(r)-1)
    let mut u = EXP_POLY_10_D;
    u = mlaf(u, f, EXP_POLY_9_D);
    u = mlaf(u, f, EXP_POLY_8_D);
    u = mlaf(u, f, EXP_POLY_7_D);
    u = mlaf(u, f, EXP_POLY_6_D);
    u = mlaf(u, f, EXP_POLY_5_D);
    u = mlaf(u, f, EXP_POLY_4_D);
    u = mlaf(u, f, EXP_POLY_3_D);
    u = mlaf(u, f, EXP_POLY_2_D);
    u = mlaf(u, f, EXP_POLY_1_D);
    let u = 1f64 + 2f64 * r / (u - r);
    let i2 = pow2i(q);
    let mut r = u * i2;
    if d < -964f64 {
        r = 0f64;
    }
    if d > 709f64 {
        r = f64::INFINITY;
    }
    r
}

#[inline]
pub fn do_exp_coeff(d: f64, coeff: &Vec<f64>) -> f64 {
    let qf = rintk(d * R_LN2);
    let q = qf as i32;

    let mut r = mlaf(qf, -L2_U, d);
    r = mlaf(qf, -L2_L, r);

    let f = r * r;
    // Poly for u = r*(exp(r)+1)/(exp(r)-1)
    let mut u = EXP_POLY_10_D;
    u = mlaf(u, f, EXP_POLY_9_D);
    u = mlaf(u, f, EXP_POLY_8_D);
    u = mlaf(u, f, EXP_POLY_7_D);
    u = mlaf(u, f, EXP_POLY_6_D);
    u = mlaf(u, f, coeff[3]);
    u = mlaf(u, f, coeff[2]);
    u = mlaf(u, f, coeff[1]);
    u = mlaf(u, f, coeff[0]);
    u = mlaf(u, f, EXP_POLY_1_D);
    let u = 1f64 + 2f64 * r / (u - r);
    let i2 = pow2i(q);
    let mut r = u * i2;
    if d < -964f64 {
        r = 0f64;
    }
    if d > 709f64 {
        r = f64::INFINITY;
    }
    r
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline]
fn do_exp_neon(d: f64) -> f64 {
    unsafe {
        let ld = vdupq_n_f64(d);
        vgetq_lane_f64::<0>(vexpq_f64(ld))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_exp_sse(d: f64) -> f64 {
    unsafe {
        let ld = _mm_set1_pd(d);
        _mm_extract_pd::<0>(_mm_exp_pd(ld))
    }
}

#[inline]
/// Computes exp with error bound *ULP 1.0*
pub fn eexp(d: f64) -> f64 {
    let mut _dispatcher: fn(f64) -> f64 = do_exp;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
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
