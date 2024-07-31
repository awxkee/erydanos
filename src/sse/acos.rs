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

use crate::{_mm_abs_pd, _mm_asin_pd, _mm_select_pd};

/// Computes arccos, error bound *ULP 2.0*
#[inline(always)]
pub unsafe fn _mm_acos_pd(x: __m128d) -> __m128d {
    let gt_zero = _mm_cmpgt_pd(x, _mm_setzero_pd());
    let x_a = _mm_abs_pd(x);
    let x_asin = _mm_asin_pd(x_a);
    let v_pi = _mm_set1_pd(std::f64::consts::FRAC_PI_2);
    return _mm_select_pd(gt_zero, _mm_sub_pd(v_pi, x_asin), _mm_add_pd(v_pi, x_asin));
}

#[cfg(test)]
mod tests {
    use crate::_mm_extract_pd;

    use super::*;

    #[test]
    fn test_acosd() {
        unsafe {
            let value = _mm_set1_pd(0.3);
            let comparison = _mm_acos_pd(value);
            let flag_1 = _mm_extract_pd::<1>(comparison);
            let control = 1.266103672779499;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm_set1_pd(-0.3);
            let comparison = _mm_acos_pd(value);
            let flag_1 = _mm_extract_pd::<1>(comparison);
            let control = 1.87548898081029412720f64;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm_set1_pd(-2f64);
            let comparison = _mm_acos_pd(value);
            let flag_1 = _mm_extract_pd::<1>(comparison);
            assert!(flag_1.is_nan());
        }
    }
}
