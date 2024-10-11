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

use crate::{
    _mm256_abs_ps, _mm256_eqzero_ps, _mm256_isinf_ps, _mm256_isnan_ps, _mm256_mlaf_ps,
    _mm256_select_ps,
};

#[inline]
/// Method that computes 2D Euclidian distance *ULP 0.6667*
pub unsafe fn _mm256_hypot_ps(x: __m256, y: __m256) -> __m256 {
    let x = _mm256_abs_ps(x);
    let y = _mm256_abs_ps(y);
    let max = _mm256_max_ps(x, y);
    let min = _mm256_min_ps(x, y);
    let r = _mm256_div_ps(min, max);
    let mut ret = _mm256_mul_ps(
        _mm256_sqrt_ps(_mm256_mlaf_ps(r, r, _mm256_set1_ps(1f32))),
        max,
    );
    let is_any_infinite = _mm256_or_ps(_mm256_isinf_ps(x), _mm256_isinf_ps(y));
    let mut is_any_nan = _mm256_or_ps(_mm256_isnan_ps(x), _mm256_isnan_ps(y));
    let is_min_zero = _mm256_eqzero_ps(min);
    is_any_nan = _mm256_or_ps(_mm256_isnan_ps(ret), is_any_nan);
    ret = _mm256_select_ps(is_any_nan, _mm256_set1_ps(f32::NAN), ret);
    ret = _mm256_select_ps(is_any_infinite, _mm256_set1_ps(f32::INFINITY), ret);
    ret = _mm256_select_ps(is_min_zero, _mm256_setzero_ps(), ret);
    ret
}

/// Method that computes 2D Euclidian distance *ULP 0.6667*, skipping Inf, Nan checks
#[inline]
pub unsafe fn _mm256_hypot_fast_ps(x: __m256, y: __m256) -> __m256 {
    let x = _mm256_abs_ps(x);
    let y = _mm256_abs_ps(y);
    let max = _mm256_max_ps(x, y);
    let min = _mm256_min_ps(x, y);
    let r = _mm256_div_ps(min, max);
    let is_min_zero = _mm256_eqzero_ps(min);
    let mut ret = _mm256_mul_ps(
        _mm256_sqrt_ps(_mm256_mlaf_ps(r, r, _mm256_set1_ps(1f32))),
        max,
    );
    ret = _mm256_select_ps(is_min_zero, _mm256_setzero_ps(), ret);
    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_mm256_extract_ps;

    #[test]
    fn test_hypot() {
        unsafe {
            // Test regular
            let vx = _mm256_set1_ps(3.);
            let vy = _mm256_set1_ps(4.);
            let comparison = _mm256_hypot_ps(vx, vy);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, 5.);
        }
    }
}
