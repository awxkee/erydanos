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

use crate::{_mm_abs_pd, _mm_eqzero_pd, _mm_isinf_pd, _mm_isnan_pd, _mm_mlaf_pd, _mm_select_pd};

#[inline]
/// Method that computes 2D Euclidian distance *ULP 0.6667*
pub unsafe fn _mm_hypot_pd(x: __m128d, y: __m128d) -> __m128d {
    let x = _mm_abs_pd(x);
    let y = _mm_abs_pd(y);
    let max = _mm_max_pd(x, y);
    let min = _mm_min_pd(x, y);
    let r = _mm_div_pd(min, max);
    let mut ret = _mm_mul_pd(_mm_sqrt_pd(_mm_mlaf_pd(r, r, _mm_set1_pd(1.))), max);
    let is_any_infinite = _mm_or_pd(_mm_isinf_pd(x), _mm_isinf_pd(y));
    let mut is_any_nan = _mm_or_pd(_mm_isnan_pd(x), _mm_isnan_pd(y));
    let is_min_zero = _mm_eqzero_pd(min);
    is_any_nan = _mm_or_pd(_mm_isnan_pd(ret), is_any_nan);
    ret = _mm_select_pd(is_any_nan, _mm_set1_pd(f64::NAN), ret);
    ret = _mm_select_pd(is_any_infinite, _mm_set1_pd(f64::INFINITY), ret);
    ret = _mm_select_pd(is_min_zero, _mm_setzero_pd(), ret);
    ret
}

/// Method that computes 2D Euclidian distance *ULP 0.6667*, skipping Inf, Nan checks
#[inline]
pub unsafe fn _mm_hypot_fast_pd(x: __m128d, y: __m128d) -> __m128d {
    let x = _mm_abs_pd(x);
    let y = _mm_abs_pd(y);
    let max = _mm_max_pd(x, y);
    let min = _mm_min_pd(x, y);
    let r = _mm_div_pd(min, max);
    let ret = _mm_mul_pd(_mm_sqrt_pd(_mm_mlaf_pd(r, r, _mm_set1_pd(1.))), max);
    ret
}

#[cfg(test)]
mod tests {
    use crate::_mm_extract_pd;

    use super::*;

    #[test]
    fn test_hypotd() {
        unsafe {
            // Test regular
            let vx = _mm_set1_pd(3.);
            let vy = _mm_set1_pd(4.);
            let comparison = _mm_hypot_pd(vx, vy);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, 5.);
        }
    }
}
