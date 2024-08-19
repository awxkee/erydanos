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

use crate::avx::generalf::_mm256_copysign_ps;
use crate::{
    _mm256_abs_ps, _mm256_exp_ps, _mm256_expq_fast_ps, _mm256_isinf_ps, _mm256_isnan_ps,
    _mm256_isneginf_ps, _mm256_isnotintegral_ps, _mm256_ln_fast_ps, _mm256_ln_ps, _mm256_select_ps,
};

#[inline]
#[target_feature(enable = "avx2")]
/// Computes pow function *ULP 2.0*
pub unsafe fn _mm256_pow_ps(d: __m256, n: __m256) -> __m256 {
    let mut c = _mm256_exp_ps(_mm256_mul_ps(n, _mm256_ln_ps(_mm256_abs_ps(d))));
    c = _mm256_copysign_ps(c, d);
    let is_infinity = _mm256_or_ps(
        _mm256_isinf_ps(d),
        _mm256_or_ps(_mm256_isinf_ps(n), _mm256_isneginf_ps(n)),
    );
    let is_power_neg_infinity = _mm256_isneginf_ps(n);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let is_nan_with_integral = _mm256_and_ps(
        _mm256_cmp_ps::<_CMP_LT_OS>(d, _mm256_setzero_ps()),
        _mm256_isnotintegral_ps(n),
    );
    let is_any_nan = _mm256_or_ps(
        _mm256_or_ps(_mm256_isnan_ps(d), _mm256_isnan_ps(n)),
        is_nan_with_integral,
    );
    let mut ret = _mm256_select_ps(is_infinity, inf, c);
    ret = _mm256_select_ps(is_power_neg_infinity, _mm256_set1_ps(0f32), ret);
    ret = _mm256_select_ps(is_any_nan, _mm256_set1_ps(f32::NAN), ret);
    ret
}

/// Method that computes pow skipping Inf, Nan checks, *ULP 2.0*
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_pow_fast_ps(d: __m256, n: __m256) -> __m256 {
    let mut c = _mm256_expq_fast_ps(_mm256_mul_ps(n, _mm256_ln_fast_ps(d)));
    c = _mm256_copysign_ps(c, d);
    c
}

#[cfg(test)]
mod tests {
    use crate::_mm256_extract_ps;

    use super::*;

    #[test]
    fn test_powf() {
        unsafe {
            // Test regular
            let value = _mm256_set1_ps(15f32);
            let power = _mm256_set1_ps(1. / 5.);
            let comparison = _mm256_pow_ps(value, power);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, 15f32.powf(1. / 5.));
        }

        unsafe {
            let value = _mm256_set1_ps(15f32);
            let power = _mm256_set1_ps(-1. / 5.);
            let comparison = _mm256_pow_ps(value, power);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            let flag_1 = flag_1.to_bits();
            let origin = 0.5818107f32.to_bits();
            let diff = flag_1.max(origin) - flag_1.min(origin);
            assert!(diff < 2);
        }

        unsafe {
            let value = _mm256_set1_ps(-15f32);
            let power = _mm256_set1_ps(1. / 5.);
            let comparison = _mm256_pow_ps(value, power);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert!(flag_1.is_nan());
        }

        unsafe {
            let value = _mm256_set1_ps(-15f32);
            let power = _mm256_set1_ps(-1. / 5.);
            let comparison = _mm256_pow_ps(value, power);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert!(flag_1.is_nan());
        }
    }
}
