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
/// Method that computes 4D Euclidian distance *ULP 0.6666*
pub unsafe fn _mm256_hypot4_pd(x: __m256d, y: __m256d, z: __m256d, w: __m256d) -> __m256d {
    let x = _mm256_abs_pd(x);
    let y = _mm256_abs_pd(y);
    let z = _mm256_abs_pd(z);
    let w = _mm256_abs_pd(w);
    let max = _mm256_max_pd(_mm256_max_pd(_mm256_max_pd(x, y), z), w);
    let norm_x = _mm256_div_pd(x, max);
    let norm_y = _mm256_div_pd(y, max);
    let norm_z = _mm256_div_pd(z, max);
    let norm_w = _mm256_div_pd(w, max);

    let accumulator = _mm256_mlaf_pd(
        norm_x,
        norm_x,
        _mm256_mlaf_pd(
            norm_y,
            norm_y,
            _mm256_mlaf_pd(norm_z, norm_z, _mm256_mul_pd(norm_w, norm_w)),
        ),
    );
    let mut ret = _mm256_mul_pd(_mm256_sqrt_pd(accumulator), max);
    let is_any_infinite = _mm256_or_pd(
        _mm256_or_pd(
            _mm256_or_pd(_mm256_isinf_pd(x), _mm256_isinf_pd(y)),
            _mm256_isinf_pd(z),
        ),
        _mm256_isinf_pd(w),
    );
    let mut is_any_nan = _mm256_or_pd(
        _mm256_or_pd(
            _mm256_or_pd(_mm256_isnan_pd(x), _mm256_isnan_pd(y)),
            _mm256_isnan_pd(z),
        ),
        _mm256_isnan_pd(w),
    );
    let is_max_zero = _mm256_cmp_pd::<_CMP_EQ_OS>(max, _mm256_setzero_pd());
    is_any_nan = _mm256_or_pd(_mm256_isnan_pd(ret), is_any_nan);
    ret = _mm256_select_pd(is_any_infinite, _mm256_set1_pd(f64::INFINITY), ret);
    ret = _mm256_select_pd(is_any_nan, _mm256_set1_pd(f64::NAN), ret);
    ret = _mm256_select_pd(is_max_zero, _mm256_set1_pd(0.), ret);
    ret
}

/// Method that computes 4D Euclidian distance *ULP 0.6666*, skipping Inf, Nan checks
#[inline]
pub unsafe fn _mm256_hypot4_fast_pd(x: __m256d, y: __m256d, z: __m256d, w: __m256d) -> __m256d {
    let x = _mm256_abs_pd(x);
    let y = _mm256_abs_pd(y);
    let z = _mm256_abs_pd(z);
    let w = _mm256_abs_pd(w);
    let max = _mm256_max_pd(_mm256_max_pd(_mm256_max_pd(x, y), z), w);
    let norm_x = _mm256_div_pd(x, max);
    let norm_y = _mm256_div_pd(y, max);
    let norm_z = _mm256_div_pd(z, max);
    let norm_w = _mm256_div_pd(w, max);

    let accumulator = _mm256_mlaf_pd(
        norm_x,
        norm_x,
        _mm256_mlaf_pd(
            norm_y,
            norm_y,
            _mm256_mlaf_pd(norm_z, norm_z, _mm256_mul_pd(norm_w, norm_w)),
        ),
    );
    let ret = _mm256_mul_pd(_mm256_sqrt_pd(accumulator), max);
    ret
}

#[cfg(test)]
mod tests {
    use crate::avx::general::_mm256_extract_pd;

    use super::*;

    #[test]
    fn test_hypot4d_avx() {
        unsafe {
            // Test regular
            let vx = _mm256_set1_pd(3.);
            let vy = _mm256_set1_pd(4.);
            let vz = _mm256_set1_pd(5.);
            let vw = _mm256_set1_pd(6.);
            let comparison = _mm256_hypot4_pd(vx, vy, vz, vw);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, _mm256_extract_pd::<1>(comparison));
            assert_eq!(flag_1, 9.27361849549570375f64);
        }

        unsafe {
            // Test regular
            let vx = _mm256_set1_pd(3.);
            let vy = _mm256_set1_pd(4.);
            let vz = _mm256_set1_pd(5.);
            let vw = _mm256_set1_pd(f64::NAN);
            let comparison = _mm256_hypot4_pd(vx, vy, vz, vw);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert!(flag_1.is_nan());
        }
    }
}
