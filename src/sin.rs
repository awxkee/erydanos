/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::generalf::{mlaf, rintk, IsNegZero};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vsinq_f64;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::{_mm_extract_pd, _mm_sin_pd};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::*;
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

pub(crate) const SIN_POLY_2_D: f64 = -0.1666666666666666666666f64;
pub(crate) const SIN_POLY_3_D: f64 = 0.008333333333333333332719f64;
pub(crate) const SIN_POLY_4_D: f64 = -0.0001984126984126984066122;
pub(crate) const SIN_POLY_5_D: f64 = 2.755731922398555921929e-6;
pub(crate) const SIN_POLY_6_D: f64 = -2.505210838533321890745e-8;
pub(crate) const SIN_POLY_7_D: f64 = 1.605904381455638364872e-10;
pub(crate) const SIN_POLY_8_D: f64 = -7.647160846222341105455e-13;
pub(crate) const SIN_POLY_9_D: f64 = 2.811227876145604544553e-15;
pub(crate) const SIN_POLY_10_D: f64 = -8.118486649859753485496e-18;

pub(crate) const PI_A2: f64 = 3.141_592_653_589_793_116;
pub(crate) const PI_B2: f64 = 1.224_646_799_147_353_207_2_e-16;

#[inline]
fn do_sin(d: f64) -> f64 {
    let qf = rintk(std::f64::consts::FRAC_1_PI * d);
    let q = qf as i64;
    let mut r = mlaf(qf, -PI_A2, d);
    r = mlaf(qf, -PI_B2, r);

    let x2 = r * r;

    if (q & 1) != 0 {
        r = -r;
    }
    let mut u = SIN_POLY_10_D;
    u = mlaf(u, x2, SIN_POLY_9_D);
    u = mlaf(u, x2, SIN_POLY_8_D);
    u = mlaf(u, x2, SIN_POLY_7_D);
    u = mlaf(u, x2, SIN_POLY_6_D);
    u = mlaf(u, x2, SIN_POLY_5_D);
    u = mlaf(u, x2, SIN_POLY_4_D);
    u = mlaf(u, x2, SIN_POLY_3_D);
    u = mlaf(u, x2, SIN_POLY_2_D);
    u = u * x2 * r + r;
    if u.isnegzero() {
        return 0.;
    }
    u
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
fn do_sin_neon(d: f64) -> f64 {
    unsafe {
        let ld = vdupq_n_f64(d);
        vgetq_lane_f64::<0>(vsinq_f64(ld))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_sin_sse(d: f64) -> f64 {
    unsafe {
        let j = _mm_set1_pd(d);
        _mm_extract_pd::<0>(_mm_sin_pd(j))
    }
}

/// Computes sine function with *ULP 1.5* on range [-15; 15]
#[inline]
pub fn esin(d: f64) -> f64 {
    let mut _dispatcher: fn(f64) -> f64 = do_sin;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
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
