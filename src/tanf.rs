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
use std::f32::consts::FRAC_2_PI;

use crate::cosf::{PI_A_F, PI_B_F, PI_C_F, PI_D_F};
use crate::generalf::{mlaf, rintfk, IsNegZero};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vtanq_f32;

pub(crate) const TAN_POLY_1_S: f32 = 0.3333353561669567628359f32;
pub(crate) const TAN_POLY_2_S: f32 = 0.1332909226735641872812f32;
pub(crate) const TAN_POLY_3_S: f32 = 0.05437330042338871738713f32;
pub(crate) const TAN_POLY_4_S: f32 = 0.01976259322538098448190f32;
pub(crate) const TAN_POLY_5_S: f32 = 0.01536309149864370613748f32;
pub(crate) const TAN_POLY_6_S: f32 = -0.008716767804671342083395f32;
pub(crate) const TAN_POLY_7_S: f32 = 0.01566058603292222557185f32;
pub(crate) const TAN_POLY_8_S: f32 = -0.008780698867440909852696f32;
pub(crate) const TAN_POLY_9_S: f32 = 0.003119367819237227984603f32;

#[inline]
fn do_tanf(d: f32) -> f32 {
    let qf = rintfk(d * FRAC_2_PI);
    let q = qf as i32;
    let mut x = mlaf(qf, -PI_A_F * 0.5, d);
    x = mlaf(qf, -PI_B_F * 0.5, x);
    x = mlaf(qf, -PI_C_F * 0.5, x);
    x = mlaf(qf, -PI_D_F * 0.5, x);

    let x2 = x * x;

    if (q & 1) != 0 {
        x = -x;
    }

    let mut u = TAN_POLY_9_S;
    u = mlaf(u, x2, TAN_POLY_8_S);
    u = mlaf(u, x2, TAN_POLY_7_S);
    u = mlaf(u, x2, TAN_POLY_6_S);
    u = mlaf(u, x2, TAN_POLY_5_S);
    u = mlaf(u, x2, TAN_POLY_4_S);
    u = mlaf(u, x2, TAN_POLY_3_S);
    u = mlaf(u, x2, TAN_POLY_2_S);
    u = mlaf(u, x2, TAN_POLY_1_S);
    u = mlaf(u, x2 * x, x);

    if (q & 1) != 0 {
        u = 1. / u;
    }

    let c = u;
    if c.isnegzero() {
        return 0f32;
    }
    c
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
fn do_tanf_neon(d: f32) -> f32 {
    unsafe {
        let ld = vdupq_n_f32(d);
        vgetq_lane_f32::<0>(vtanq_f32(ld))
    }
}

/// Computes tan *ULP 2.0*
#[inline]
pub fn etanf(d: f32) -> f32 {
    let mut _dispatcher: fn(f32) -> f32 = do_tanf;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_tanf_neon;
    }
    _dispatcher(d)
}
