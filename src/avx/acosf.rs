/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{_mm256_abs_ps, _mm256_asin_ps, _mm256_select_ps};

/// Computes arccos, error bound *ULP 2.0*
#[inline(always)]
pub unsafe fn _mm256_acos_ps(x: __m256) -> __m256 {
    let gt_zero = _mm256_cmp_ps::<_CMP_GT_OS>(x, _mm256_setzero_ps());
    let x_a = _mm256_abs_ps(x);
    let x_asin = _mm256_asin_ps(x_a);
    let v_pi = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
    return _mm256_select_ps(
        gt_zero,
        _mm256_sub_ps(v_pi, x_asin),
        _mm256_add_ps(v_pi, x_asin),
    );
}

#[cfg(test)]
mod tests {
    use crate::_mm256_extract_ps;

    use super::*;

    #[test]
    fn test_acosf() {
        unsafe {
            let value = _mm256_set1_ps(0.3);
            let comparison = _mm256_acos_ps(value);
            let flag_1 = _mm256_extract_ps::<1>(comparison);
            let control = 1.2661037;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm256_set1_ps(-0.3);
            let comparison = _mm256_acos_ps(value);
            let flag_1 = _mm256_extract_ps::<1>(comparison);
            let control = 1.87548898081029412720f32;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm256_set1_ps(-2f32);
            let comparison = _mm256_acos_ps(value);
            let flag_1 = _mm256_extract_ps::<1>(comparison);
            assert!(flag_1.is_nan());
        }
    }
}
