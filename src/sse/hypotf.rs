/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::{_mm_abs_ps, _mm_eqzero_ps, _mm_isinf_ps, _mm_isnan_ps, _mm_mlaf_ps, _mm_select_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
/// Method that computes 2D Euclidian distance *ULP 0.5*
pub unsafe fn _mm_hypot_ps(x: __m128, y: __m128) -> __m128 {
    let x = _mm_abs_ps(x);
    let y = _mm_abs_ps(y);
    let max = _mm_max_ps(x, y);
    let min = _mm_min_ps(x, y);
    let r = _mm_div_ps(min, max);
    let mut ret = _mm_mul_ps(_mm_sqrt_ps(_mm_mlaf_ps(r, r, _mm_set1_ps(1f32))), max);
    let is_any_infinite = _mm_and_ps(_mm_isinf_ps(x), _mm_isinf_ps(y));
    let is_any_nan = _mm_and_ps(_mm_isnan_ps(x), _mm_isnan_ps(y));
    let is_min_zero = _mm_eqzero_ps(min);
    let is_result_nan = _mm_isnan_ps(ret);
    ret = _mm_select_ps(is_any_infinite, _mm_set1_ps(f32::INFINITY), ret);
    ret = _mm_select_ps(is_any_nan, _mm_set1_ps(f32::NAN), ret);
    ret = _mm_select_ps(is_min_zero, _mm_set1_ps(0f32), ret);
    ret = _mm_select_ps(is_result_nan, _mm_set1_ps(f32::INFINITY), ret);
    ret
}

/// Method that computes 2D Euclidian distance *ULP 0.5*, skipping Inf, Nan checks
#[inline]
pub unsafe fn _mm_hypot_fast_ps(x: __m128, y: __m128) -> __m128 {
    let y = _mm_abs_ps(y);
    let max = _mm_max_ps(x, y);
    let min = _mm_min_ps(x, y);
    let r = _mm_div_ps(min, max);
    let ret = _mm_mul_ps(_mm_sqrt_ps(_mm_mlaf_ps(r, r, _mm_set1_ps(1f32))), max);
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypot() {
        unsafe {
            // Test regular
            let vx = _mm_set1_ps(3.);
            let vy = _mm_set1_ps(4.);
            let comparison = _mm_hypot_ps(vx, vy);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, 5.);
        }
    }
}
