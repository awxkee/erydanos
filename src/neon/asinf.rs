/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::asinf::{
    ASIN_POLY_2_F, ASIN_POLY_3_F, ASIN_POLY_4_F, ASIN_POLY_5_F, ASIN_POLY_6_F, ASIN_POLY_7_F,
    ASIN_POLY_8_F, ASIN_POLY_9_F,
};
use crate::neon::general::{vcopysignq_f32, vmlafq_f32};
use std::arch::aarch64::*;

#[inline]
pub unsafe fn vasinq_f32(d: float32x4_t) -> float32x4_t {
    let ones = vdupq_n_f32(1f32);
    let ca = vabsq_f32(d);
    let nan_mask = vcgtq_f32(ca, ones);
    // for more 0.5
    let reverse_05_mask = vcgeq_f32(ca, vdupq_n_f32(0.5f32));
    let reversed = vsqrtq_f32(vdivq_f32(vsubq_f32(ones, ca), vdupq_n_f32(2f32)));
    let x = vbslq_f32(reverse_05_mask, reversed, ca);
    let zeros_is_zeros = vceqzq_f32(d);
    let mut u = vreinterpretq_f32_u32(vdupq_n_u32(ASIN_POLY_9_F));
    u = vmlafq_f32(u, x, vreinterpretq_f32_u32(vdupq_n_u32(ASIN_POLY_8_F)));
    u = vmlafq_f32(u, x, vreinterpretq_f32_u32(vdupq_n_u32(ASIN_POLY_7_F)));
    u = vmlafq_f32(u, x, vreinterpretq_f32_u32(vdupq_n_u32(ASIN_POLY_6_F)));
    u = vmlafq_f32(u, x, vreinterpretq_f32_u32(vdupq_n_u32(ASIN_POLY_5_F)));
    u = vmlafq_f32(u, x, vreinterpretq_f32_u32(vdupq_n_u32(ASIN_POLY_4_F)));
    u = vmlafq_f32(u, x, vreinterpretq_f32_u32(vdupq_n_u32(ASIN_POLY_3_F)));
    u = vmlafq_f32(u, x, vreinterpretq_f32_u32(vdupq_n_u32(ASIN_POLY_2_F)));
    u = vmulq_f32(u, x);
    let j = u;
    let reconstruct_reversed = vmlafq_f32(
        vdupq_n_f32(-2f32),
        j,
        vdupq_n_f32(std::f32::consts::FRAC_PI_2),
    );
    let mut ret = vbslq_f32(reverse_05_mask, reconstruct_reversed, j);
    ret = vbslq_f32(nan_mask, vdupq_n_f32(f32::NAN), ret);
    ret = vbslq_f32(zeros_is_zeros, vdupq_n_f32(0f32), ret);
    vcopysignq_f32(ret, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asinf() {
        unsafe {
            let value = vdupq_n_f32(0.3);
            let comparison = vasinq_f32(value);
            let flag_1 = vgetq_lane_f32::<1>(comparison);
            let control = 0.304692654015397507972f32;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = vdupq_n_f32(-0.3);
            let comparison = vasinq_f32(value);
            let flag_1 = vgetq_lane_f32::<1>(comparison);
            let control = -0.304692654015397507972f32;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = vdupq_n_f32(-2f32);
            let comparison = vasinq_f32(value);
            let flag_1 = vgetq_lane_f32::<1>(comparison);
            assert!(flag_1.is_nan());
        }
    }
}
