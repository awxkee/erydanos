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
    _mm256_eqzero_ps, _mm256_isinf_ps, _mm256_isneginf_ps, _mm256_mlaf_ps, _mm256_mul_epu64,
    _mm256_packts_epi64, _mm256_select_ps,
};

#[inline(always)]
unsafe fn halley_cbrt(x: __m256, a: __m256) -> __m256 {
    let tx = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
    let twos = _mm256_set1_ps(2f32);
    let num = _mm256_mlaf_ps(twos, a, tx);
    let den = _mm256_mlaf_ps(twos, tx, a);
    let scale = _mm256_div_ps(num, den);
    _mm256_mul_ps(x, scale)
}

#[inline(always)]
unsafe fn integer_pow_1_3(hx: __m256i) -> __m256i {
    let scale = _mm256_set1_epi64x(341);
    let hi = _mm256_srli_epi64::<10>(_mm256_mul_epu64(
        _mm256_unpackhi_epi32(hx, _mm256_setzero_si256()),
        scale,
    ));
    let lo = _mm256_srli_epi64::<10>(_mm256_mul_epu64(
        _mm256_unpacklo_epi32(hx, _mm256_setzero_si256()),
        scale,
    ));
    _mm256_packts_epi64(lo, hi)
}

/// Takes cube root from value *ULP 1.5*, Skipping NaN, Inf checks
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cbrt_fast_ps(x: __m256) -> __m256 {
    let mut ui = _mm256_castps_si256(x);
    let hx = _mm256_and_si256(ui, _mm256_set1_epi32(0x7fffffff));

    let hx = _mm256_add_epi32(integer_pow_1_3(hx), _mm256_set1_epi32(709958130));

    #[allow(overflowing_literals)]
    let m = _mm256_set1_epi32(0x80000000);
    ui = _mm256_and_si256(ui, m);
    ui = _mm256_or_si256(ui, hx);

    let t = _mm256_castsi256_ps(ui);

    let c0 = halley_cbrt(t, x);
    let c1 = halley_cbrt(c0, x);
    let v = _mm256_select_ps(_mm256_eqzero_ps(x), _mm256_set1_ps(0f32), c1);
    v
}

/// Takes cube root from value *ULP 1.5*
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_cbrt_ps(x: __m256) -> __m256 {
    let c1 = _mm256_cbrt_fast_ps(x);
    let mut v = _mm256_select_ps(_mm256_isinf_ps(x), _mm256_set1_ps(f32::INFINITY), c1);
    v = _mm256_select_ps(_mm256_isneginf_ps(x), _mm256_set1_ps(f32::NEG_INFINITY), v);
    v
}

#[cfg(test)]
mod tests {
    use crate::_mm256_extract_ps;

    use super::*;

    #[test]
    fn test_cbrtf() {
        unsafe {
            // Test regular
            let value = _mm256_set1_ps(0.0201934222);
            let comparison = _mm256_cbrt_ps(value);
            let flag_1 = _mm256_extract_ps::<3>(comparison);
            assert_eq!(flag_1, 0.272313982f32);
        }

        unsafe {
            // Test regular
            let value = _mm256_set1_ps(0.0201934222);
            let comparison = _mm256_cbrt_ps(value);
            let flag_1 = _mm256_extract_ps::<3>(comparison);
            assert_eq!(flag_1, 0.272313982f32);
        }

        unsafe {
            // Test regular
            let value = _mm256_set1_ps(27f32);
            let comparison = _mm256_cbrt_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, 3f32);
        }

        unsafe {
            // Test regular
            let value = _mm256_set1_ps(0.5f32);
            let comparison = _mm256_cbrt_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, 0.7937005f32);
        }

        unsafe {
            // Test Infinity
            let value = _mm256_set1_ps(f32::INFINITY);
            let comparison = _mm256_cbrt_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, f32::INFINITY);
        }

        unsafe {
            // Test Neg Infinity
            let value = _mm256_set1_ps(-27f32);
            let comparison = _mm256_cbrt_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, -3f32);
        }

        unsafe {
            let value = _mm256_set1_ps(f32::NEG_INFINITY);
            let comparison = _mm256_cbrt_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, f32::NEG_INFINITY);
        }
    }
}
