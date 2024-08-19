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

use crate::sse::poly128::{_mm_addw_epi128, _mm_movn_epi128, _mm_mull_epu64, _mm_srli_epi128x};
use crate::{_mm_eqzero_pd, _mm_isinf_pd, _mm_isneginf_pd, _mm_mlaf_pd, _mm_select_pd};

#[inline(always)]
unsafe fn halley_cbrt(x: __m128d, a: __m128d) -> __m128d {
    let tx = _mm_mul_pd(_mm_mul_pd(x, x), x);
    let twos = _mm_set1_pd(2.);
    let num = _mm_mlaf_pd(twos, a, tx);
    let den = _mm_mlaf_pd(twos, tx, a);
    let scale = _mm_div_pd(num, den);
    _mm_mul_pd(x, scale)
}

#[inline(always)]
unsafe fn integer_pow_1_3(hx: __m128i) -> __m128i {
    let scale = _mm_set1_epi64x(341);
    let wide = _mm_mull_epu64(hx, scale);
    let shifted = _mm_srli_epi128x::<10>(wide);
    let addiction = _mm_set1_epi64x(715094163);
    let product = _mm_addw_epi128(shifted, addiction);
    _mm_movn_epi128(product)
}

/// Takes cube root from value *ULP 1.5*, Skipping NaN, Inf checks
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cbrt_fast_pd(x: __m128d) -> __m128d {
    let mut ui = _mm_castpd_si128(x);
    let hx = _mm_and_si128(_mm_srli_epi64::<32>(ui), _mm_set1_epi64x(0x7fffffff));

    let hx = integer_pow_1_3(hx);

    #[allow(overflowing_literals)]
    let m = _mm_set1_epi64x(1 << 63);
    ui = _mm_and_si128(ui, m);
    ui = _mm_or_si128(ui, _mm_slli_epi64::<32>(hx));

    let t = _mm_castsi128_pd(ui);

    let c0 = halley_cbrt(t, x);
    let c1 = halley_cbrt(c0, x);
    let c2 = halley_cbrt(c1, x);
    let v = _mm_select_pd(_mm_eqzero_pd(x), _mm_set1_pd(0.), c2);
    v
}

/// Takes cube root from value *ULP 1.5*
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_cbrt_pd(x: __m128d) -> __m128d {
    let c1 = _mm_cbrt_fast_pd(x);
    let mut v = _mm_select_pd(_mm_isinf_pd(x), _mm_set1_pd(f64::INFINITY), c1);
    v = _mm_select_pd(_mm_isneginf_pd(x), _mm_set1_pd(f64::NEG_INFINITY), v);
    v
}

#[cfg(test)]
mod tests {
    use crate::_mm_extract_pd;

    use super::*;

    #[test]
    fn test_cbrtd() {
        unsafe {
            // Test regular
            let value = _mm_set1_pd(0.0201934222);
            let comparison = _mm_cbrt_pd(value);
            let flag_1 = _mm_extract_pd::<1>(comparison);
            assert_eq!(flag_1, 0.27231400353631246186777917307287727f64);
        }

        unsafe {
            // Test regular
            let value = _mm_set1_pd(0.0201934222);
            let comparison = _mm_cbrt_fast_pd(value);
            let flag_1 = _mm_extract_pd::<1>(comparison);
            assert_eq!(flag_1, 0.27231400353631246186777917307287727f64);
        }

        unsafe {
            // Test regular
            let value = _mm_set1_pd(27f64);
            let comparison = _mm_cbrt_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, 3f64);
        }

        unsafe {
            // Test regular
            let value = _mm_set1_pd(0.5);
            let comparison = _mm_cbrt_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, 0.793700525984099737375f64);
        }

        unsafe {
            // Test big
            let value = _mm_set1_pd(150000000f64);
            let comparison = _mm_cbrt_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, 531.32928459130553302387f64);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_pd(f64::INFINITY);
            let comparison = _mm_cbrt_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, f64::INFINITY);
        }

        unsafe {
            // Test Neg Infinity
            let value = _mm_set1_pd(-27f64);
            let comparison = _mm_cbrt_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, -3f64);
        }

        unsafe {
            let value = _mm_set1_pd(f64::NEG_INFINITY);
            let comparison = _mm_cbrt_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, f64::NEG_INFINITY);
        }
    }
}
