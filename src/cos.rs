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

use crate::generalf::{mlaf, rintk, IsNegZero};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vcosq_f64;
use crate::sin::{
    PI_A2, PI_B2, SIN_POLY_10_D, SIN_POLY_2_D, SIN_POLY_3_D, SIN_POLY_4_D, SIN_POLY_5_D,
    SIN_POLY_6_D, SIN_POLY_7_D, SIN_POLY_8_D, SIN_POLY_9_D,
};

#[inline]
fn do_cos(d: f64) -> f64 {
    let qf = 1. + 2. * rintk(std::f64::consts::FRAC_1_PI * d - 0.5);
    let q = qf as i64;
    let mut r = mlaf(qf, -PI_A2 * 0.5, d);
    r = mlaf(qf, -PI_B2 * 0.5, r);

    let x2 = r * r;

    if q & 2 == 0 {
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
fn do_cos_neon(d: f64) -> f64 {
    unsafe {
        let j = vdupq_n_f64(d);
        vgetq_lane_f64::<0>(vcosq_f64(j))
    }
}

/// Cosine function
///
/// The error bound of the returned value is `1.5 ULP` on range [-15; 15]
pub fn ecos(d: f64) -> f64 {
    let mut _dispatcher: fn(f64) -> f64 = do_cos;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_cos_neon;
    }
    _dispatcher(d)
}
