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
#[target_feature(enable = "sse4.1")]
/// Method that computes 3D Euclidian distance *ULP 0.66667*
pub unsafe fn _mm_hypot3_pd(x: __m128d, y: __m128d, z: __m128d) -> __m128d {
    let x = _mm_abs_pd(x);
    let y = _mm_abs_pd(y);
    let z = _mm_abs_pd(z);
    let max = _mm_max_pd(_mm_max_pd(x, y), z);
    let recip_max = _mm_div_pd(_mm_set1_pd(1.), max);
    let norm_x = _mm_mul_pd(x, recip_max);
    let norm_y = _mm_mul_pd(y, recip_max);
    let norm_z = _mm_mul_pd(z, recip_max);

    let accumulator = _mm_mlaf_pd(
        norm_x,
        norm_x,
        _mm_mlaf_pd(norm_y, norm_y, _mm_mul_pd(norm_z, norm_z)),
    );
    let mut ret = _mm_mul_pd(_mm_sqrt_pd(accumulator), max);
    let is_any_infinite = _mm_or_pd(_mm_or_pd(_mm_isinf_pd(x), _mm_isinf_pd(y)), _mm_isinf_pd(z));
    let mut is_any_nan = _mm_or_pd(_mm_or_pd(_mm_isnan_pd(x), _mm_isnan_pd(y)), _mm_isnan_pd(z));
    let is_max_zero = _mm_eqzero_pd(max);
    is_any_nan = _mm_or_pd(_mm_isnan_pd(ret), is_any_nan);
    ret = _mm_select_pd(is_any_nan, _mm_set1_pd(f64::NAN), ret);
    ret = _mm_select_pd(is_any_infinite, _mm_set1_pd(f64::INFINITY), ret);
    ret = _mm_select_pd(is_max_zero, _mm_setzero_pd(), ret);
    ret
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Method that computes 3D Euclidian distance *ULP 0.66667*, skipping Inf, Nan checks
pub unsafe fn _mm_hypot3_fast_pd(x: __m128d, y: __m128d, z: __m128d) -> __m128d {
    let x = _mm_abs_pd(x);
    let y = _mm_abs_pd(y);
    let z = _mm_abs_pd(z);
    let max = _mm_max_pd(_mm_max_pd(x, y), z);
    let recip_max = _mm_div_pd(_mm_set1_pd(1.), max);
    let norm_x = _mm_mul_pd(x, recip_max);
    let norm_y = _mm_mul_pd(y, recip_max);
    let norm_z = _mm_mul_pd(z, recip_max);

    let accumulator = _mm_mlaf_pd(
        norm_x,
        norm_x,
        _mm_mlaf_pd(norm_y, norm_y, _mm_mul_pd(norm_z, norm_z)),
    );
    let ret = _mm_mul_pd(_mm_sqrt_pd(accumulator), max);
    ret
}

#[cfg(test)]
mod tests {
    use crate::_mm_extract_pd;

    use super::*;

    #[test]
    fn test_hypot3d_sse() {
        unsafe {
            // Test regular
            let vx = _mm_set1_pd(3.);
            let vy = _mm_set1_pd(4.);
            let vz = _mm_set1_pd(5.);
            let comparison = _mm_hypot3_pd(vx, vy, vz);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, _mm_extract_pd::<1>(comparison));
            assert_eq!(flag_1, 7.0710678118654752440f64);
        }

        unsafe {
            // Test fast
            let vx = _mm_set1_pd(3.);
            let vy = _mm_set1_pd(4.);
            let vz = _mm_set1_pd(5.);
            let comparison = _mm_hypot3_fast_pd(vx, vy, vz);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, _mm_extract_pd::<1>(comparison));
            assert_eq!(flag_1, 7.0710678118654752440f64);
        }
    }
}
