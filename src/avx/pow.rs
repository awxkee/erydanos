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

use crate::avx::general::_mm256_isnotintegral_pd;
use crate::{
    _mm256_abs_pd, _mm256_copysign_pd, _mm256_exp_pd, _mm256_expq_fast_pd, _mm256_isinf_pd,
    _mm256_isnan_pd, _mm256_isneginf_pd, _mm256_ln_fast_pd, _mm256_ln_pd, _mm256_select_pd,
};

#[inline]
/// Computes pow function *ULP 2.0*
pub unsafe fn _mm256_pow_pd(d: __m256d, n: __m256d) -> __m256d {
    let mut c = _mm256_exp_pd(_mm256_mul_pd(n, _mm256_ln_pd(_mm256_abs_pd(d))));
    c = _mm256_copysign_pd(c, d);
    let is_infinity = _mm256_or_pd(
        _mm256_isinf_pd(d),
        _mm256_or_pd(_mm256_isinf_pd(n), _mm256_isneginf_pd(n)),
    );
    let is_power_neg_infinity = _mm256_isneginf_pd(n);
    // Not integral values do not allowed for negative numbers
    let inf = _mm256_set1_pd(f64::INFINITY);
    let is_nan_with_integral = _mm256_and_pd(
        _mm256_cmp_pd::<_CMP_LT_OS>(d, _mm256_setzero_pd()),
        _mm256_isnotintegral_pd(n),
    );
    let is_any_nan = _mm256_or_pd(
        _mm256_or_pd(_mm256_isnan_pd(d), _mm256_isnan_pd(n)),
        is_nan_with_integral,
    );
    let mut ret = _mm256_select_pd(is_infinity, inf, c);
    ret = _mm256_select_pd(is_power_neg_infinity, _mm256_set1_pd(0.), ret);
    ret = _mm256_select_pd(is_any_nan, _mm256_set1_pd(f64::NAN), ret);
    ret
}

/// Method that computes pow skipping Inf, Nan checks, *ULP 2.0*
#[inline]
pub unsafe fn _mm256_pow_fast_pd(d: __m256d, n: __m256d) -> __m256d {
    let mut c = _mm256_expq_fast_pd(_mm256_mul_pd(n, _mm256_ln_fast_pd(d)));
    c = _mm256_copysign_pd(c, d);
    c
}

#[cfg(test)]
mod tests {
    use crate::avx::general::_mm256_extract_pd;

    use super::*;

    #[test]
    fn test_powd() {
        unsafe {
            // Test regular
            let value = _mm256_set1_pd(15f64);
            let power = _mm256_set1_pd(1. / 5.);
            let comparison = _mm256_pow_pd(value, power);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, 1.7187719275874787f64);
        }

        unsafe {
            let value = _mm256_set1_pd(15f64);
            let power = _mm256_set1_pd(-1. / 5.);
            let comparison = _mm256_pow_pd(value, power);
            let rs = _mm256_extract_pd::<0>(comparison);
            let flag_1 = rs.to_bits();
            // Rust returns NAN for negative values with < 1 power, this is not correct
            let origin = 0.581810759152688049901847f64.to_bits();
            let diff = flag_1.max(origin) - flag_1.min(origin);
            assert!(diff < 2);
        }

        unsafe {
            let value = _mm256_set1_pd(-15f64);
            let power = _mm256_set1_pd(1. / 5.);
            let comparison = _mm256_pow_pd(value, power);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert!(flag_1.is_nan());
        }

        unsafe {
            let value = _mm256_set1_pd(-15f64);
            let power = _mm256_set1_pd(-1. / 5.);
            let comparison = _mm256_pow_pd(value, power);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert!(flag_1.is_nan());
        }

        unsafe {
            let value = _mm256_set1_pd(-2f64);
            let power = _mm256_set1_pd(3.);
            let comparison = _mm256_pow_pd(value, power);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, -7.999999999999998);
        }
    }
}
