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
pub unsafe fn _mm_hypot3_ps(x: __m128, y: __m128, z: __m128) -> __m128 {
    let x = _mm_abs_ps(x);
    let y = _mm_abs_ps(y);
    let z = _mm_abs_ps(z);
    let max = _mm_max_ps(_mm_max_ps(x, y), z);
    let norm_x = _mm_div_ps(x, max);
    let norm_y = _mm_div_ps(y, max);
    let norm_z = _mm_div_ps(z, max);

    let accumulator = _mm_mlaf_ps(
        norm_x,
        norm_x,
        _mm_mlaf_ps(norm_y, norm_y, _mm_mul_ps(norm_z, norm_z)),
    );
    let mut ret = _mm_mul_ps(_mm_sqrt_ps(accumulator), max);
    let is_any_infinite = _mm_or_ps(_mm_or_ps(_mm_isinf_ps(x), _mm_isinf_ps(y)), _mm_isinf_ps(z));
    let is_any_nan = _mm_or_ps(_mm_isnan_ps(x), _mm_isnan_ps(y));
    let is_max_zero = _mm_eqzero_ps(max);
    let is_result_nan = _mm_isnan_ps(ret);
    ret = _mm_select_ps(is_any_infinite, _mm_set1_ps(f32::INFINITY), ret);
    ret = _mm_select_ps(is_any_nan, _mm_set1_ps(f32::NAN), ret);
    ret = _mm_select_ps(is_max_zero, _mm_set1_ps(0f32), ret);
    ret = _mm_select_ps(is_result_nan, _mm_set1_ps(f32::INFINITY), ret);
    ret
}

/// Method that computes 3D Euclidian distance *ULP 0.6666*, skipping Inf, Nan checks
#[inline]
pub unsafe fn _mm_hypot3_fast_ps(x: __m128, y: __m128, z: __m128) -> __m128 {
    let x = _mm_abs_ps(x);
    let y = _mm_abs_ps(y);
    let z = _mm_abs_ps(z);
    let max = _mm_max_ps(_mm_max_ps(x, y), z);
    let norm_x = _mm_div_ps(x, max);
    let norm_y = _mm_div_ps(y, max);
    let norm_z = _mm_div_ps(z, max);
    let accumulator = _mm_mlaf_ps(
        norm_x,
        norm_x,
        _mm_mlaf_ps(norm_y, norm_y, _mm_mul_ps(norm_z, norm_z)),
    );
    let ret = _mm_mul_ps(_mm_sqrt_ps(accumulator), max);
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypot3f_sse() {
        unsafe {
            // Test regular
            let vx = _mm_set1_ps(3.);
            let vy = _mm_set1_ps(4.);
            let vz = _mm_set1_ps(5.);
            let comparison = _mm_hypot3_ps(vx, vy, vz);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, f32::from_bits(_mm_extract_ps::<1>(comparison) as u32));
            assert_eq!(flag_1, f32::from_bits(_mm_extract_ps::<2>(comparison) as u32));
            assert_eq!(flag_1, f32::from_bits(_mm_extract_ps::<3>(comparison) as u32));
            assert_eq!(flag_1, 7.0710678118654752440f32);
        }
    }
}
