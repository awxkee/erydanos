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
/// Method that computes 3D Euclidian distance *ULP 0.66667*
pub unsafe fn _mm256_hypot3_pd(x: __m256d, y: __m256d, z: __m256d) -> __m256d {
    let x = _mm256_abs_pd(x);
    let y = _mm256_abs_pd(y);
    let z = _mm256_abs_pd(z);
    let max = _mm256_max_pd(_mm256_max_pd(x, y), z);
    let norm_x = _mm256_div_pd(x, max);
    let norm_y = _mm256_div_pd(y, max);
    let norm_z = _mm256_div_pd(z, max);

    let accumulator = _mm256_mlaf_pd(
        norm_x,
        norm_x,
        _mm256_mlaf_pd(norm_y, norm_y, _mm256_mul_pd(norm_z, norm_z)),
    );
    let mut ret = _mm256_mul_pd(_mm256_sqrt_pd(accumulator), max);
    let is_any_infinite = _mm256_or_pd(
        _mm256_or_pd(_mm256_isinf_pd(x), _mm256_isinf_pd(y)),
        _mm256_isinf_pd(z),
    );
    let mut is_any_nan = _mm256_or_pd(
        _mm256_or_pd(_mm256_isnan_pd(x), _mm256_isnan_pd(y)),
        _mm256_isnan_pd(z),
    );
    let is_max_zero = _mm256_cmp_pd::<_CMP_EQ_OS>(max, _mm256_setzero_pd());
    is_any_nan = _mm256_or_pd(_mm256_isnan_pd(ret), is_any_nan);
    ret = _mm256_select_pd(is_any_infinite, _mm256_set1_pd(f64::INFINITY), ret);
    ret = _mm256_select_pd(is_any_nan, _mm256_set1_pd(f64::NAN), ret);
    ret = _mm256_select_pd(is_max_zero, _mm256_set1_pd(0.), ret);
    ret
}

/// Method that computes 3D Euclidian distance *ULP 0.66667*, skipping Inf, Nan checks
#[inline]
pub unsafe fn _mm256_hypot3_fast_pd(x: __m256d, y: __m256d, z: __m256d) -> __m256d {
    let x = _mm256_abs_pd(x);
    let y = _mm256_abs_pd(y);
    let z = _mm256_abs_pd(z);
    let max = _mm256_max_pd(_mm256_max_pd(x, y), z);
    let norm_x = _mm256_div_pd(x, max);
    let norm_y = _mm256_div_pd(y, max);
    let norm_z = _mm256_div_pd(z, max);

    let accumulator = _mm256_mlaf_pd(
        norm_x,
        norm_x,
        _mm256_mlaf_pd(norm_y, norm_y, _mm256_mul_pd(norm_z, norm_z)),
    );
    let ret = _mm256_mul_pd(_mm256_sqrt_pd(accumulator), max);
    ret
}

#[cfg(test)]
mod tests {
    use crate::avx::general::_mm256_extract_pd;

    use super::*;

    #[test]
    fn test_hypot3d_avx() {
        unsafe {
            // Test regular
            let vx = _mm256_set1_pd(3.);
            let vy = _mm256_set1_pd(4.);
            let vz = _mm256_set1_pd(5.);
            let comparison = _mm256_hypot3_pd(vx, vy, vz);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, _mm256_extract_pd::<1>(comparison));
            assert_eq!(flag_1, 7.0710678118654752440f64);
        }

        unsafe {
            // Test fast
            let vx = _mm256_set1_pd(3.);
            let vy = _mm256_set1_pd(4.);
            let vz = _mm256_set1_pd(5.);
            let comparison = _mm256_hypot3_fast_pd(vx, vy, vz);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, _mm256_extract_pd::<1>(comparison));
            assert_eq!(flag_1, 7.0710678118654752440f64);
        }
    }
}
