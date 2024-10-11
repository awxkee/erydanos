/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::{
    _mm256_abs_ps, _mm256_eqzero_ps, _mm256_isinf_ps, _mm256_isnan_ps, _mm256_mlaf_ps,
    _mm256_select_ps,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
/// Method that computes 3D Euclidian distance *ULP 0.6667*
pub unsafe fn _mm256_hypot3_ps(x: __m256, y: __m256, z: __m256) -> __m256 {
    let x = _mm256_abs_ps(x);
    let y = _mm256_abs_ps(y);
    let z = _mm256_abs_ps(z);
    let max = _mm256_max_ps(_mm256_max_ps(x, y), z);
    let recip_max = _mm256_div_ps(_mm256_set1_ps(1.), max);
    let norm_x = _mm256_mul_ps(x, recip_max);
    let norm_y = _mm256_mul_ps(y, recip_max);
    let norm_z = _mm256_mul_ps(z, recip_max);

    let accumulator = _mm256_mlaf_ps(
        norm_x,
        norm_x,
        _mm256_mlaf_ps(norm_y, norm_y, _mm256_mul_ps(norm_z, norm_z)),
    );
    let mut ret = _mm256_mul_ps(_mm256_sqrt_ps(accumulator), max);
    let is_any_infinite = _mm256_or_ps(
        _mm256_or_ps(_mm256_isinf_ps(x), _mm256_isinf_ps(y)),
        _mm256_isinf_ps(z),
    );
    let mut is_any_nan = _mm256_or_ps(
        _mm256_or_ps(_mm256_isnan_ps(x), _mm256_isnan_ps(y)),
        _mm256_isnan_ps(z),
    );
    let is_max_zero = _mm256_eqzero_ps(max);
    is_any_nan = _mm256_or_ps(_mm256_isnan_ps(ret), is_any_nan);
    ret = _mm256_select_ps(is_any_nan, _mm256_set1_ps(f32::NAN), ret);
    ret = _mm256_select_ps(is_any_infinite, _mm256_set1_ps(f32::INFINITY), ret);
    ret = _mm256_select_ps(is_max_zero, _mm256_setzero_ps(), ret);
    ret
}

/// Method that computes 3D Euclidian distance *ULP 0.6666*, skipping Inf, Nan checks
#[inline]
pub unsafe fn _mm256_hypot3_fast_ps(x: __m256, y: __m256, z: __m256) -> __m256 {
    let x = _mm256_abs_ps(x);
    let y = _mm256_abs_ps(y);
    let z = _mm256_abs_ps(z);
    let max = _mm256_max_ps(_mm256_max_ps(x, y), z);
    let recip_max = _mm256_div_ps(_mm256_set1_ps(1.), max);
    let norm_x = _mm256_mul_ps(x, recip_max);
    let norm_y = _mm256_mul_ps(y, recip_max);
    let norm_z = _mm256_mul_ps(z, recip_max);

    let is_max_zero = _mm256_eqzero_ps(max);

    let accumulator = _mm256_mlaf_ps(
        norm_x,
        norm_x,
        _mm256_mlaf_ps(norm_y, norm_y, _mm256_mul_ps(norm_z, norm_z)),
    );
    let mut ret = _mm256_mul_ps(_mm256_sqrt_ps(accumulator), max);
    ret = _mm256_select_ps(is_max_zero, _mm256_setzero_ps(), ret);
    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_mm256_extract_ps;

    #[test]
    fn test_hypot3f_avx() {
        unsafe {
            // Test regular
            let vx = _mm256_set1_ps(3.);
            let vy = _mm256_set1_ps(4.);
            let vz = _mm256_set1_ps(5.);
            let comparison = _mm256_hypot3_ps(vx, vy, vz);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, _mm256_extract_ps::<1>(comparison));
            assert_eq!(flag_1, _mm256_extract_ps::<2>(comparison));
            assert_eq!(flag_1, _mm256_extract_ps::<3>(comparison));
            assert_eq!(flag_1, 7.0710678118654752440f32);
        }
    }
}
