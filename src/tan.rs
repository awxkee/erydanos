/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::generalf::{mlaf, rintk};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vtanq_f64;
use crate::sin::{PI_A2, PI_B2};
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::{_mm_extract_pd, _mm_tan_pd};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::{vdupq_n_f64, vgetq_lane_f64};
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

pub(crate) const TAN_POLY_1_D: f64 = 0.333_333_333_333_334_369_5;
pub(crate) const TAN_POLY_2_D: f64 = 0.133_333_333_333_050_058_1;
pub(crate) const TAN_POLY_3_D: f64 = 0.539_682_539_951_727_297_e-1;
pub(crate) const TAN_POLY_4_D: f64 = 0.218_694_872_818_553_549_8_e-1;
pub(crate) const TAN_POLY_5_D: f64 = 0.886_326_840_956_311_312_6_e-2;
pub(crate) const TAN_POLY_6_D: f64 = 0.359_161_154_079_249_951_9_e-2;
pub(crate) const TAN_POLY_7_D: f64 = 0.146_078_150_240_278_449_4_e-2;
pub(crate) const TAN_POLY_8_D: f64 = 0.561_921_973_811_432_373_5_e-3;
pub(crate) const TAN_POLY_9_D: f64 = 0.324_509_882_663_927_631_6_e-3;

#[inline]
fn do_tan(d: f64) -> f64 {
    let qlf = rintk(d * std::f64::consts::FRAC_2_PI);
    let q = qlf as isize;
    let mut x = mlaf(qlf, -PI_A2 * 0.5, d);
    x = mlaf(qlf, -PI_B2 * 0.5, x);

    if (q & 1) != 0 {
        x = -x;
    }

    x *= 0.5;

    let x2 = x * x;
    let mut u = TAN_POLY_9_D;
    u = mlaf(u, x2, TAN_POLY_8_D);
    u = mlaf(u, x2, TAN_POLY_7_D);
    u = mlaf(u, x2, TAN_POLY_6_D);
    u = mlaf(u, x2, TAN_POLY_5_D);
    u = mlaf(u, x2, TAN_POLY_4_D);
    u = mlaf(u, x2, TAN_POLY_3_D);
    u = mlaf(u, x2, TAN_POLY_2_D);
    u = mlaf(u, x2, TAN_POLY_1_D);
    u = mlaf(u, x2 * x, x);

    let mut c = 2. * u / (1. - u * u);

    if (q & 1) != 0 {
        c = 1. / c;
    }

    c
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
fn do_tan_neon(d: f64) -> f64 {
    unsafe {
        let ld = vdupq_n_f64(d);
        vgetq_lane_f64::<0>(vtanq_f64(ld))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline(always)]
fn do_tan_sse(d: f64) -> f64 {
    unsafe {
        let ld = _mm_set1_pd(d);
        _mm_extract_pd::<0>(_mm_tan_pd(ld))
    }
}

#[inline]
/// Computes tan with error bound *ULP 2.0*
pub fn etan(d: f64) -> f64 {
    let mut _dispatcher: fn(f64) -> f64 = do_tan;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_tan_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_tan_sse;
    }
    _dispatcher(d)
}
