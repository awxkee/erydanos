/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use std::arch::aarch64::*;

use crate::vasinq_f32;

/// Computes arccos, error bound *ULP 2.0*
#[inline]
pub unsafe fn vacosq_f32(x: float32x4_t) -> float32x4_t {
    let gt_zero = vcgtzq_f32(x);
    let x_a = vabsq_f32(x);
    let x_asin = vasinq_f32(x_a);
    let v_pi = vdupq_n_f32(std::f32::consts::FRAC_PI_2);
    return vbslq_f32(gt_zero, vsubq_f32(v_pi, x_asin), vaddq_f32(v_pi, x_asin));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acosf() {
        unsafe {
            let value = vdupq_n_f32(0.3);
            let comparison = vacosq_f32(value);
            let flag_1 = vgetq_lane_f32::<1>(comparison);
            let control = 1.2661037;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = vdupq_n_f32(-0.3);
            let comparison = vacosq_f32(value);
            let flag_1 = vgetq_lane_f32::<1>(comparison);
            let control = 1.87548898081029412720f32;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = vdupq_n_f32(-2f32);
            let comparison = vacosq_f32(value);
            let flag_1 = vgetq_lane_f32::<1>(comparison);
            assert!(flag_1.is_nan());
        }
    }
}
