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
use std::arch::aarch64::{vdupq_n_f64, vgetq_lane_f64};

use crate::generalf::{ilogb2k, ldexp3k, mlaf};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vlnq_f64;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::{_mm_extract_pd, _mm_ln_pd};
#[cfg(all(target_feature = "sse4.1", target_arch = "x86"))]
use std::arch::x86::*;
#[cfg(all(target_feature = "sse4.1", target_arch = "x86_64"))]
use std::arch::x86_64::*;

pub(crate) const LN_POLY_1_D: f64 = 2.;
pub(crate) const LN_POLY_2_D: f64 = 0.666_666_666_666_777_874_006_3;
pub(crate) const LN_POLY_3_D: f64 = 0.399_999_999_950_799_600_689_777;
pub(crate) const LN_POLY_4_D: f64 = 0.285_714_294_746_548_025_383_248;
pub(crate) const LN_POLY_5_D: f64 = 0.222_221_366_518_767_365_905_163;
pub(crate) const LN_POLY_6_D: f64 = 0.181_863_266_251_982_985_677_316;
pub(crate) const LN_POLY_7_D: f64 = 0.152_519_917_006_351_951_593_857;
pub(crate) const LN_POLY_8_D: f64 = 0.153_487_338_491_425_068_243_146;

// Absolute error 1.136351756823757474514312*10^-18
#[inline]
fn do_ln(d: f64) -> f64 {
    // ln(𝑥)=ln(𝑎)+𝑛ln(2)
    let n = ilogb2k(d * (1. / 0.75));
    let a = ldexp3k(d, -n);

    let x = (a - 1.) / (a + 1.);
    let x2 = x * x;
    let mut u = LN_POLY_8_D;
    u = mlaf(u, x2, LN_POLY_7_D);
    u = mlaf(u, x2, LN_POLY_6_D);
    u = mlaf(u, x2, LN_POLY_5_D);
    u = mlaf(u, x2, LN_POLY_4_D);
    u = mlaf(u, x2, LN_POLY_3_D);
    u = mlaf(u, x2, LN_POLY_2_D);
    u = mlaf(u, x2, LN_POLY_1_D);

    return if d == 0f64 {
        f64::NEG_INFINITY
    } else if (d < 0.) || d.is_nan() {
        f64::NAN
    } else if d.is_infinite() {
        f64::INFINITY
    } else {
        x * u + std::f64::consts::LN_2 * (n as f64)
    };
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline]
fn do_ln_neon(x: f64) -> f64 {
    unsafe {
        let vx = vdupq_n_f64(x);
        vgetq_lane_f64::<0>(vlnq_f64(vx))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_ln_sse(x: f64) -> f64 {
    unsafe {
        let vx = _mm_set1_pd(x);
        _mm_extract_pd::<0>(_mm_ln_pd(vx))
    }
}

/// Computes natural logarithm *ULP 3.5*
pub fn eln(d: f64) -> f64 {
    let mut _dispatcher: fn(f64) -> f64 = do_ln;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_ln_neon;
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
