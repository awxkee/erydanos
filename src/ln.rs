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
use crate::neon::ln::vlnq_f64;

pub(crate) const LN_POLY_1_D: f64 = 1.9999999999999999977273f64;
pub(crate) const LN_POLY_2_D: f64 = 0.66666666666667652664208f64;
pub(crate) const LN_POLY_3_D: f64 = 0.39999999999298481856961f64;
pub(crate) const LN_POLY_4_D: f64 = 0.2857142876143736275867f64;
pub(crate) const LN_POLY_5_D: f64 = 0.22222196978570449848180f64;
pub(crate) const LN_POLY_6_D: f64 = 0.18183635619969267855036f64;
pub(crate) const LN_POLY_7_D: f64 = 0.14810676790894580143625f64;
pub(crate) const LN_POLY_8_D: f64 = 0.15312429691862720905483f64;

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

pub fn eln(d: f64) -> f64 {
    let mut _dispatcher: fn(f64) -> f64 = do_ln;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_ln_neon;
    }
    _dispatcher(d)
}
