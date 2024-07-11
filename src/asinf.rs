/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::abs::eabsf;
use crate::generalf::{copysignfk, mlaf};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vasinq_f32;
use crate::sqrtf::esqrtf;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};

pub(crate) const ASIN_POLY_2_F: u32 = 0x3f7ffffe; // 0.99999996772155
pub(crate) const ASIN_POLY_3_F: u32 = 0x36f8402e; // 7.398621160063735e-6
pub(crate) const ASIN_POLY_4_F: u32 = 0x3e2a61e2; // 0.1663871446825855
pub(crate) const ASIN_POLY_5_F: u32 = 0x3b82e304; // 3.994731408211294e-3
pub(crate) const ASIN_POLY_6_F: u32 = 0x3d4218ba; // 4.739361033517798e-2
pub(crate) const ASIN_POLY_7_F: u32 = 0x3dce28d2; // 0.10064352904194375
pub(crate) const ASIN_POLY_8_F: u32 = 0xbe16c000; // -0.14719234953315127
pub(crate) const ASIN_POLY_9_F: u32 = 0x3e292000; // 0.16517876838808404

/**
Best ULP 1.9091235399246216
Best Coefficients [1.0, 7.3982183e-6, 0.1663863, 0.003995185, 0.047398783, 0.100653425]
*/

#[inline]
fn do_asinf(c: f32) -> f32 {
    if eabsf(c) > 1f32 {
        return f32::NAN;
    }
    if c == 0. {
        return 0.;
    }
    let mut q = 0;
    let ca = eabsf(c);
    let d = if ca >= 0.5f32 {
        q = 1;
        esqrtf((1f32 - ca) / 2f32)
    } else {
        ca
    };
    let x = d;
    let mut u = f32::from_bits(ASIN_POLY_9_F);
    u = mlaf(u, x, f32::from_bits(ASIN_POLY_8_F));
    u = mlaf(u, x, f32::from_bits(ASIN_POLY_7_F));
    u = mlaf(u, x, f32::from_bits(ASIN_POLY_6_F));
    u = mlaf(u, x, f32::from_bits(ASIN_POLY_5_F));
    u = mlaf(u, x, f32::from_bits(ASIN_POLY_4_F));
    u = mlaf(u, x, f32::from_bits(ASIN_POLY_3_F));
    u = mlaf(u, x, f32::from_bits(ASIN_POLY_2_F));
    u = u * x;
    let j = u;
    let z = if q & 1 != 0 {
        std::f32::consts::FRAC_PI_2 - 2f32 * j
    } else {
        j
    };
    copysignfk(z, c)
}

#[inline]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
fn do_asinf_neon(d: f32) -> f32 {
    unsafe {
        let ld = vdupq_n_f32(d);
        vgetq_lane_f32::<0>(vasinq_f32(ld))
    }
}

/// Computes asin for an argument *ULP 2.0*
#[inline]
pub fn easinf(d: f32) -> f32 {
    let mut _dispatcher: fn(f32) -> f32 = do_asinf;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_asinf_neon;
    }
    _dispatcher(d)
}
