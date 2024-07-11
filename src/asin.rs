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

use crate::abs::eabs;
use crate::generalf::{copysignk, mlaf};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vasinq_f64;
use crate::sqrt::esqrt;

pub const ASIN_POLY_1_D: f64 = 1.000000000000000023366;
pub const ASIN_POLY_2_D: f64 = -3.176610416071895664242e-14;
pub const ASIN_POLY_3_D: f64 = 0.1666666666740305207309;
pub const ASIN_POLY_4_D: f64 = -6.792754203042129818201e-10;
pub(crate) const ASIN_POLY_5_D: f64 = 0.07500003310580183667325;
pub(crate) const ASIN_POLY_6_D: f64 = -9.831045152232444057386e-7;
pub(crate) const ASIN_POLY_7_D: f64 = 0.04466221059932533189613;
pub(crate) const ASIN_POLY_8_D: f64 = -0.0002666335211728438820425;
pub(crate) const ASIN_POLY_9_D: f64 = 0.03304751490956488865548;
pub(crate) const ASIN_POLY_10_D: f64 = -0.01981071126900658252673;
pub(crate) const ASIN_POLY_11_D: f64 = 0.1335232659628717297626;
pub(crate) const ASIN_POLY_12_D: f64 = -0.4746006107797438986858;
pub(crate) const ASIN_POLY_13_D: f64 = 1.561435324832807339641;
pub(crate) const ASIN_POLY_14_D: f64 = -3.804159966909373578153;
pub(crate) const ASIN_POLY_15_D: f64 = 7.002267933032339176679;
pub(crate) const ASIN_POLY_16_D: f64 = -9.296807687094489647642;
pub(crate) const ASIN_POLY_17_D: f64 = 8.504557880477015298817;
pub(crate) const ASIN_POLY_18_D: f64 = -4.795580928637641353576;
pub(crate) const ASIN_POLY_19_D: f64 = 1.287092068615565929449;

#[inline]
fn do_asin(c: f64) -> f64 {
    if eabs(c) > 1f64 {
        return f64::NAN;
    }
    let mut q = 0;
    let ca = eabs(c);
    let d = if ca >= 0.5f64 {
        q = 1;
        esqrt((1f64 - ca) / 2f64)
    } else {
        ca
    };
    let x = d;
    let mut u = ASIN_POLY_19_D;
    u = mlaf(u, x, ASIN_POLY_18_D);
    u = mlaf(u, x, ASIN_POLY_17_D);
    u = mlaf(u, x, ASIN_POLY_16_D);
    u = mlaf(u, x, ASIN_POLY_15_D);
    u = mlaf(u, x, ASIN_POLY_14_D);
    u = mlaf(u, x, ASIN_POLY_13_D);
    u = mlaf(u, x, ASIN_POLY_12_D);
    u = mlaf(u, x, ASIN_POLY_11_D);
    u = mlaf(u, x, ASIN_POLY_10_D);
    u = mlaf(u, x, ASIN_POLY_9_D);
    u = mlaf(u, x, ASIN_POLY_8_D);
    u = mlaf(u, x, ASIN_POLY_7_D);
    u = mlaf(u, x, ASIN_POLY_6_D);
    u = mlaf(u, x, ASIN_POLY_5_D);
    u = mlaf(u, x, ASIN_POLY_4_D);
    u = mlaf(u, x, ASIN_POLY_3_D);
    u = mlaf(u, x, ASIN_POLY_2_D);
    u = mlaf(u, x, ASIN_POLY_1_D);
    u = u * x;
    let v = if q & 1 != 0 {
        std::f64::consts::FRAC_PI_2 - 2f64 * u
    } else {
        u
    };
    copysignk(v, c)
}

#[inline]
pub fn do_asin_coeffs(c: f64, coeffs: &Vec<f64>) -> f64 {
    if eabs(c) > 1f64 {
        return f64::NAN;
    }
    let mut q = 0;
    let ca = eabs(c);
    let d = if ca >= 0.5f64 {
        q = 1;
        esqrt((1f64 - ca) / 2f64)
    } else {
        ca
    };
    let x = d;
    let mut u = ASIN_POLY_19_D;
    u = mlaf(u, x, ASIN_POLY_18_D);
    u = mlaf(u, x, ASIN_POLY_17_D);
    u = mlaf(u, x, ASIN_POLY_16_D);
    u = mlaf(u, x, ASIN_POLY_15_D);
    u = mlaf(u, x, ASIN_POLY_14_D);
    u = mlaf(u, x, ASIN_POLY_13_D);
    u = mlaf(u, x, ASIN_POLY_12_D);
    u = mlaf(u, x, ASIN_POLY_11_D);
    u = mlaf(u, x, ASIN_POLY_10_D);
    u = mlaf(u, x, ASIN_POLY_9_D);
    u = mlaf(u, x, ASIN_POLY_8_D);
    u = mlaf(u, x, ASIN_POLY_7_D);
    u = mlaf(u, x, ASIN_POLY_6_D);
    u = mlaf(u, x, ASIN_POLY_5_D);
    u = mlaf(u, x, coeffs[3]);
    u = mlaf(u, x, coeffs[2]);
    u = mlaf(u, x, coeffs[1]);
    u = mlaf(u, x, coeffs[0]);
    u = u * x;
    let v = if q & 1 != 0 {
        std::f64::consts::FRAC_PI_2 - 2f64 * u
    } else {
        u
    };
    copysignk(v, c)
}

#[inline]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
fn do_asin_neon(d: f64) -> f64 {
    unsafe {
        let ld = vdupq_n_f64(d);
        vgetq_lane_f64::<0>(vasinq_f64(ld))
    }
}

#[inline]
/// Computes asin with *ULP 1.5*
pub fn easin(d: f64) -> f64 {
    let mut _dispatcher: fn(f64) -> f64 = do_asin;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_asin_neon;
    }
    _dispatcher(d)
}
