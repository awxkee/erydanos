/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

use crate::vasinq_f64;

/// Computes arccos, error bound *ULP 2.0*
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn vacosq_f64(x: float64x2_t) -> float64x2_t {
    let gt_zero = vcgtzq_f64(x);
    let x_a = vabsq_f64(x);
    let x_asin = vasinq_f64(x_a);
    let v_pi = vdupq_n_f64(std::f64::consts::FRAC_PI_2);
    return vbslq_f64(gt_zero, vsubq_f64(v_pi, x_asin), vaddq_f64(v_pi, x_asin));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acosd() {
        unsafe {
            let value = vdupq_n_f64(0.3);
            let comparison = vacosq_f64(value);
            let flag_1 = vgetq_lane_f64::<1>(comparison);
            let control = 1.266103672779499f64;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = vdupq_n_f64(-0.3);
            let comparison = vacosq_f64(value);
            let flag_1 = vgetq_lane_f64::<1>(comparison);
            let control = 1.87548898081029412720f64;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = vdupq_n_f64(-2f64);
            let comparison = vacosq_f64(value);
            let flag_1 = vgetq_lane_f64::<1>(comparison);
            assert!(flag_1.is_nan());
        }
    }
}
