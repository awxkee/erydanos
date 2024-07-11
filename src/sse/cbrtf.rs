/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::{
    _mm_eqzero_ps, _mm_isinf_ps, _mm_isneginf_ps, _mm_mlaf_ps, _mm_mul_epu64, _mm_packus_epi64,
    _mm_select_ps,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn halley_cbrt(x: __m128, a: __m128) -> __m128 {
    let tx = _mm_mul_ps(_mm_mul_ps(x, x), x);
    let twos = _mm_set1_ps(2f32);
    let num = _mm_mlaf_ps(twos, a, tx);
    let den = _mm_mlaf_ps(twos, tx, a);
    let scale = _mm_div_ps(num, den);
    _mm_mul_ps(x, scale)
}

#[inline(always)]
unsafe fn integer_pow_1_3(hx: __m128i) -> __m128i {
    let scale = _mm_set1_epi32(341);
    let hi = _mm_srli_epi64::<10>(_mm_mul_epu64(
        _mm_unpackhi_epi32(hx, _mm_setzero_si128()),
        scale,
    ));
    let lo = _mm_srli_epi64::<10>(_mm_mul_epu64(
        _mm_unpacklo_epi32(hx, _mm_setzero_si128()),
        _mm_set1_epi64x(341),
    ));
    _mm_packus_epi64(lo, hi)
}

/// Takes cube root from value *ULP 1.5*
#[inline(always)]
pub unsafe fn _mm_cbrt_ps(x: __m128) -> __m128 {
    let mut ui = _mm_castps_si128(x);
    let hx = _mm_and_si128(ui, _mm_set1_epi32(0x7fffffff));

    let hx = _mm_add_epi32(integer_pow_1_3(hx), _mm_set1_epi32(709958130));

    #[allow(overflowing_literals)]
    let m = _mm_set1_epi32(0x80000000);
    ui = _mm_and_si128(ui, m);
    ui = _mm_or_si128(ui, hx);

    let t = _mm_castsi128_ps(ui);

    let c0 = halley_cbrt(t, x);
    let c1 = halley_cbrt(c0, x);
    let mut v = _mm_select_ps(_mm_eqzero_ps(x), _mm_set1_ps(0f32), c1);
    v = _mm_select_ps(_mm_isinf_ps(x), _mm_set1_ps(f32::INFINITY), v);
    v = _mm_select_ps(_mm_isneginf_ps(x), _mm_set1_ps(f32::NEG_INFINITY), v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expf() {
        unsafe {
            // Test regular
            let value = _mm_set1_ps(27f32);
            let comparison = _mm_cbrt_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, 3f32);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_ps(f32::INFINITY);
            let comparison = _mm_cbrt_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, f32::INFINITY);
        }

        unsafe {
            // Test Neg Infinity
            let value = _mm_set1_ps(-27f32);
            let comparison = _mm_cbrt_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, -3f32);
        }

        unsafe {
            let value = _mm_set1_ps(f32::NEG_INFINITY);
            let comparison = _mm_cbrt_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, f32::NEG_INFINITY);
        }
    }
}
