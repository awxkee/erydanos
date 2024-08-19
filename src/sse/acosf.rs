/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::{_mm_abs_ps, _mm_asin_ps, _mm_select_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes arccos, error bound *ULP 2.0*
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_acos_ps(x: __m128) -> __m128 {
    let gt_zero = _mm_cmpgt_ps(x, _mm_setzero_ps());
    let x_a = _mm_abs_ps(x);
    let x_asin = _mm_asin_ps(x_a);
    let v_pi = _mm_set1_ps(std::f32::consts::FRAC_PI_2);
    return _mm_select_ps(gt_zero, _mm_sub_ps(v_pi, x_asin), _mm_add_ps(v_pi, x_asin));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acosf() {
        unsafe {
            let value = _mm_set1_ps(0.3);
            let comparison = _mm_acos_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<1>(comparison) as u32);
            let control = 1.2661037;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm_set1_ps(-0.3);
            let comparison = _mm_acos_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<1>(comparison) as u32);
            let control = 1.87548898081029412720f32;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm_set1_ps(-2f32);
            let comparison = _mm_acos_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<1>(comparison) as u32);
            assert!(flag_1.is_nan());
        }
    }
}
