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
    _mm_abs_ps, _mm_copysign_ps, _mm_exp_fast_ps, _mm_exp_ps, _mm_isinf_ps, _mm_isnan_ps,
    _mm_isneginf_ps, _mm_isnotintegral_ps, _mm_ln_fast_ps, _mm_ln_ps, _mm_select_ps,
};

#[inline]
#[target_feature(enable = "sse4.1")]
/// Computes pow function *ULP 2.0*
pub unsafe fn _mm_pow_ps(d: __m128, n: __m128) -> __m128 {
    let mut c = _mm_exp_ps(_mm_mul_ps(n, _mm_ln_ps(_mm_abs_ps(d))));
    c = _mm_copysign_ps(c, d);
    let is_infinity = _mm_or_ps(
        _mm_isinf_ps(d),
        _mm_or_ps(_mm_isinf_ps(n), _mm_isneginf_ps(n)),
    );
    let is_power_neg_infinity = _mm_isneginf_ps(n);
    let inf = _mm_set1_ps(f32::INFINITY);
    let is_nan_with_integral =
        _mm_and_ps(_mm_cmplt_ps(d, _mm_setzero_ps()), _mm_isnotintegral_ps(n));
    let is_any_nan = _mm_or_ps(
        _mm_or_ps(_mm_isnan_ps(d), _mm_isnan_ps(n)),
        is_nan_with_integral,
    );
    let mut ret = _mm_select_ps(is_infinity, inf, c);
    ret = _mm_select_ps(is_power_neg_infinity, _mm_set1_ps(0f32), ret);
    ret = _mm_select_ps(is_any_nan, _mm_set1_ps(f32::NAN), ret);
    ret
}

/// Method that computes pow skipping Inf, Nan checks, *ULP 2.0*
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_pow_fast_ps(d: __m128, n: __m128) -> __m128 {
    let mut c = _mm_exp_fast_ps(_mm_mul_ps(n, _mm_ln_fast_ps(d)));
    c = _mm_copysign_ps(c, d);
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powf() {
        unsafe {
            // Test regular
            let value = _mm_set1_ps(15f32);
            let power = _mm_set1_ps(1. / 5.);
            let comparison = _mm_pow_ps(value, power);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, 15f32.powf(1. / 5.));
        }

        unsafe {
            let value = _mm_set1_ps(15f32);
            let power = _mm_set1_ps(-1. / 5.);
            let comparison = _mm_pow_ps(value, power);
            let rs = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            let flag_1 = rs.to_bits();
            let origin = 0.5818107f32.to_bits();
            let diff = flag_1.max(origin) - flag_1.min(origin);
            assert!(diff < 2);
        }

        unsafe {
            let value = _mm_set1_ps(-15f32);
            let power = _mm_set1_ps(1. / 5.);
            let comparison = _mm_pow_ps(value, power);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert!(flag_1.is_nan());
        }

        unsafe {
            let value = _mm_set1_ps(-15f32);
            let power = _mm_set1_ps(-1. / 5.);
            let comparison = _mm_pow_ps(value, power);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert!(flag_1.is_nan());
        }
    }
}
