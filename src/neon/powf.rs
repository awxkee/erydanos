/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::expf::{vexpq_f32, vexpq_fast_f32};
use crate::neon::general::{vcopysignq_f32, visinfq_f32, visnanq_f32, visneginfq_f32};
use crate::neon::lnf::{vlnq_f32, vlnq_fast_f32};
use std::arch::aarch64::*;
use crate::visnotintegralq_f32;

#[inline]
/// Computes pow function *ULP 2.0*
pub unsafe fn vpowq_f32(d: float32x4_t, n: float32x4_t) -> float32x4_t {
    let mut c = vexpq_f32(vmulq_f32(n, vlnq_f32(vabsq_f32(d))));
    c = vcopysignq_f32(c, d);
    let is_infinity = vorrq_u32(visinfq_f32(d), vorrq_u32(visinfq_f32(n), visneginfq_f32(n)));
    let is_power_neg_infinity = visneginfq_f32(n);
    let is_not_neg_integral = vandq_u32(vcltzq_f32(d), visnotintegralq_f32(n));
    let is_any_nan = vorrq_u32(vorrq_u32(visnanq_f32(d), visnanq_f32(n)), is_not_neg_integral);
    let mut ret = vbslq_f32(is_infinity, vdupq_n_f32(f32::INFINITY), c);
    ret = vbslq_f32(is_power_neg_infinity, vdupq_n_f32(0f32), ret);
    ret = vbslq_f32(is_any_nan, vdupq_n_f32(f32::NAN), ret);
    ret
}

/// Method that computes pow skipping Inf, Nan checks, *ULP 2.0*
#[inline]
pub unsafe fn vpowq_fast_f32(d: float32x4_t, n: float32x4_t) -> float32x4_t {
    let mut c = vexpq_fast_f32(vmulq_f32(n, vlnq_fast_f32(d)));
    c = vcopysignq_f32(c, d);
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powf() {
        unsafe {
            // Test regular
            let value = vdupq_n_f32(15f32);
            let power = vdupq_n_f32(1. / 5.);
            let comparison = vpowq_f32(value, power);
            let flag_1 = vgetq_lane_f32::<0>(comparison);
            assert_eq!(flag_1, 15f32.powf(1. / 5.));
        }

        unsafe {
            let value = vdupq_n_f32(15f32);
            let power = vdupq_n_f32(-1. / 5.);
            let comparison = vpowq_f32(value, power);
            let rs = vgetq_lane_f32::<0>(comparison);
            let flag_1 = rs.to_bits();
            // Rust returns NAN for negative values with < 1 power, this is not correct
            let origin = 0.5818107f32.to_bits();
            let diff = flag_1.max(origin) - flag_1.min(origin);
            assert!(diff < 2);
        }

        unsafe {
            let value = vdupq_n_f32(-15f32);
            let power = vdupq_n_f32(1. / 5.);
            let comparison = vpowq_f32(value, power);
            let flag_1 = vgetq_lane_f32::<0>(comparison);
            assert!(flag_1.is_nan());
        }

        unsafe {
            let value = vdupq_n_f32(-15f32);
            let power = vdupq_n_f32(-1. / 5.);
            let comparison = vpowq_f32(value, power);
            let flag_1 = vgetq_lane_f32::<0>(comparison);
            assert!(flag_1.is_nan());
        }
    }
}
