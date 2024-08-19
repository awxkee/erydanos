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

use crate::{_mm256_abs_pd, _mm256_isinf_pd, _mm256_isnan_pd, _mm256_mlaf_pd, _mm256_select_pd};

#[inline]
#[target_feature(enable = "avx2")]
/// Method that computes 2D Euclidian distance *ULP 0.6667*
pub unsafe fn _mm256_hypot_pd(x: __m256d, y: __m256d) -> __m256d {
    let x = _mm256_abs_pd(x);
    let y = _mm256_abs_pd(y);
    let max = _mm256_max_pd(x, y);
    let min = _mm256_min_pd(x, y);
    let r = _mm256_div_pd(min, max);
    let mut ret = _mm256_mul_pd(
        _mm256_sqrt_pd(_mm256_mlaf_pd(r, r, _mm256_set1_pd(1.))),
        max,
    );
    let is_any_infinite = _mm256_or_pd(_mm256_isinf_pd(x), _mm256_isinf_pd(y));
    let mut is_any_nan = _mm256_or_pd(_mm256_isnan_pd(x), _mm256_isnan_pd(y));
    let is_min_zero = _mm256_cmp_pd::<_CMP_EQ_OS>(min, _mm256_setzero_pd());
    is_any_nan = _mm256_or_pd(_mm256_isnan_pd(ret), is_any_nan);
    ret = _mm256_select_pd(is_any_nan, _mm256_set1_pd(f64::NAN), ret);
    ret = _mm256_select_pd(is_any_infinite, _mm256_set1_pd(f64::INFINITY), ret);
    ret = _mm256_select_pd(is_min_zero, _mm256_setzero_pd(), ret);
    ret
}

/// Method that computes 2D Euclidian distance *ULP 0.6667*, skipping Inf, Nan checks
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_hypot_fast_pd(x: __m256d, y: __m256d) -> __m256d {
    let x = _mm256_abs_pd(x);
    let y = _mm256_abs_pd(y);
    let max = _mm256_max_pd(x, y);
    let min = _mm256_min_pd(x, y);
    let r = _mm256_div_pd(min, max);
    let ret = _mm256_mul_pd(
        _mm256_sqrt_pd(_mm256_mlaf_pd(r, r, _mm256_set1_pd(1.))),
        max,
    );
    ret
}

#[cfg(test)]
mod tests {
    use crate::avx::general::_mm256_extract_pd;

    use super::*;

    #[test]
    fn test_hypotd() {
        unsafe {
            // Test regular
            let vx = _mm256_set1_pd(3.);
            let vy = _mm256_set1_pd(4.);
            let comparison = _mm256_hypot_pd(vx, vy);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, 5.);
        }
    }
}
