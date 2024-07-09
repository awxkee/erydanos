/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::generalf::mlaf;

// Chebyshev
pub const ATAN_POLY_1_D: f64 = 0.9999999999999999f64;
pub const ATAN_POLY_2_D: f64 = -0.33333333333330667f64;
pub const ATAN_POLY_3_D: f64 = 0.19999999999752202f64;
pub const ATAN_POLY_4_D: f64 = -0.14285714274686542f64;
pub const ATAN_POLY_5_D: f64 = 0.1111111082752676523817;
pub const ATAN_POLY_6_D: f64 = -0.09090904383293700958748;
pub const ATAN_POLY_7_D: f64 = 0.07692253564182837240753;
pub const ATAN_POLY_8_D: f64 = -0.06666214370915691620596;
pub const ATAN_POLY_9_D: f64 = 0.05879509976336718711646;
pub const ATAN_POLY_10_D: f64 = -0.05249366638684718970866;
pub const ATAN_POLY_11_D: f64 = 0.04709246978012999772633;
pub const ATAN_POLY_12_D: f64 = -0.04187077253614157915467;
pub const ATAN_POLY_13_D: f64 = 0.03602449800199764785191;
pub const ATAN_POLY_14_D: f64 = -0.02897361824350472260206;
pub const ATAN_POLY_15_D: f64 = 0.02089260664889144716522;
pub const ATAN_POLY_16_D: f64 = -0.01291264672794562707631;
pub const ATAN_POLY_17_D: f64 = 0.006522051887574912863745;
pub const ATAN_POLY_18_D: f64 = -0.002548984741415465180309;
pub const ATAN_POLY_19_D: f64 = 0.0007161859397322825641862;
pub const ATAN_POLY_20_D: f64 = -0.0001278956066478230268751;
pub const ATAN_POLY_21_D: f64 = 0.00001085532590549307282752;

#[inline]
fn do_atan(d: f64) -> f64 {
    let mut x = d;
    let q = if x < 0f64 {
        x = -x;
        1
    } else {
        0
    };
    let c = x;
    if x > 1f64 {
        x = 1f64 / x;
    }

    let x2 = x * x;
    let mut u = ATAN_POLY_21_D;
    u = mlaf(u, x2, ATAN_POLY_20_D);
    u = mlaf(u, x2, ATAN_POLY_19_D);
    u = mlaf(u, x2, ATAN_POLY_18_D);
    u = mlaf(u, x2, ATAN_POLY_17_D);
    u = mlaf(u, x2, ATAN_POLY_16_D);
    u = mlaf(u, x2, ATAN_POLY_15_D);
    u = mlaf(u, x2, ATAN_POLY_14_D);
    u = mlaf(u, x2, ATAN_POLY_13_D);
    u = mlaf(u, x2, ATAN_POLY_12_D);
    u = mlaf(u, x2, ATAN_POLY_11_D);
    u = mlaf(u, x2, ATAN_POLY_10_D);
    u = mlaf(u, x2, ATAN_POLY_9_D);
    u = mlaf(u, x2, ATAN_POLY_8_D);
    u = mlaf(u, x2, ATAN_POLY_7_D);
    u = mlaf(u, x2, ATAN_POLY_6_D);
    u = mlaf(u, x2, ATAN_POLY_5_D);
    u = mlaf(u, x2, ATAN_POLY_4_D);
    u = mlaf(u, x2, ATAN_POLY_3_D);
    u = mlaf(u, x2, ATAN_POLY_2_D);
    u = mlaf(u, x2, ATAN_POLY_1_D);
    u = u * x;
    u = if c > 1f64 {
        std::f64::consts::FRAC_PI_2 - u
    } else {
        u
    };
    if q & 1 != 0 {
        u = -u;
    }
    u
}

#[inline]
pub fn do_atan_coeffs(d: f64, coeffs: &Vec<f64>) -> f64 {
    let mut x = d;
    let q = if x < 0f64 {
        x = -x;
        1
    } else {
        0
    };
    let c = x;
    if x > 1f64 {
        x = 1f64 / x;
    }

    let x2 = x * x;
    let mut u = ATAN_POLY_21_D;
    u = mlaf(u, x2, ATAN_POLY_20_D);
    u = mlaf(u, x2, ATAN_POLY_19_D);
    u = mlaf(u, x2, ATAN_POLY_18_D);
    u = mlaf(u, x2, ATAN_POLY_17_D);
    u = mlaf(u, x2, ATAN_POLY_16_D);
    u = mlaf(u, x2, ATAN_POLY_15_D);
    u = mlaf(u, x2, ATAN_POLY_14_D);
    u = mlaf(u, x2, ATAN_POLY_13_D);
    u = mlaf(u, x2, ATAN_POLY_12_D);
    u = mlaf(u, x2, ATAN_POLY_11_D);
    u = mlaf(u, x2, ATAN_POLY_10_D);
    u = mlaf(u, x2, ATAN_POLY_9_D);
    u = mlaf(u, x2, ATAN_POLY_8_D);
    u = mlaf(u, x2, ATAN_POLY_7_D);
    u = mlaf(u, x2, ATAN_POLY_6_D);
    u = mlaf(u, x2, ATAN_POLY_5_D);
    u = mlaf(u, x2, ATAN_POLY_4_D);
    u = mlaf(u, x2, coeffs[2]);
    u = mlaf(u, x2, coeffs[1]);
    u = mlaf(u, x2, coeffs[0]);
    u = u * x;
    u = if c > 1f64 {
        std::f64::consts::FRAC_PI_2 - u
    } else {
        u
    };
    if q & 1 != 0 {
        u = -u;
    }
    u
}

#[inline]
/// Computes atan for f64 with error bound *ULP 2.0*
pub fn eatan(d: f64) -> f64 {
    let mut _dispatcher: fn(f64) -> f64 = do_atan;
    _dispatcher(d)
}
