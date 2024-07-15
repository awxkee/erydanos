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

#[inline(always)]
/// Founds n in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn _mm_ilogb2k_pd(d: __m128d) -> __m128i {
    _mm_sub_epi64(
        _mm_and_si128(
            _mm_srli_epi64::<52>(_mm_castpd_si128(d)),
            _mm_set1_epi64x(0x7ff),
        ),
        _mm_set1_epi64x(0x3ff),
    )
}

#[inline(always)]
/// Founds a in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn _mm_ldexp3k_pd(x: __m128d, n: __m128i) -> __m128d {
    _mm_castsi128_pd(_mm_add_epi64(_mm_castpd_si128(x), _mm_slli_epi64::<52>(n)))
}

#[inline(always)]
/// Computes a*b + c
pub unsafe fn _mm_mlaf_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    _mm_prefer_fma_pd(c, b, a)
}

#[cfg(not(target_feature = "fma"))]
#[inline]
/// Computes b*c + a using fma when available
pub unsafe fn _mm_prefer_fma_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    return _mm_add_pd(_mm_mul_pd(b, c), a);
}

#[cfg(target_feature = "fma")]
#[inline]
/// Computes b*c + a using fma when available
pub unsafe fn _mm_prefer_fma_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    return _mm_fmadd_pd(b, c, a);
}

#[inline(always)]
/// Modulus operator for f64
pub unsafe fn _mm_abs_pd(f: __m128d) -> __m128d {
    return _mm_castsi128_pd(_mm_andnot_si128(
        _mm_castpd_si128(_mm_set1_pd(-0.0f64)),
        _mm_castpd_si128(f),
    ));
}

#[inline(always)]
/// Negates value
pub unsafe fn _mm_neg_pd(f: __m128d) -> __m128d {
    _mm_sub_pd(_mm_set1_pd(0.), f)
}

#[inline(always)]
/// Returns flag value is Infinity
pub unsafe fn _mm_isinf_pd(d: __m128d) -> __m128d {
    return _mm_cmpeq_pd(_mm_abs_pd(d), _mm_set1_pd(f64::INFINITY));
}

#[inline(always)]
/// Extracts f64 value
pub unsafe fn _mm_extract_pd<const IMM: i32>(d: __m128d) -> f64 {
    f64::from_bits(_mm_extract_epi64::<IMM>(_mm_castpd_si128(d)) as u64)
}

#[inline(always)]
/// Returns true flag if value is NaN
pub unsafe fn _mm_isnan_pd(d: __m128d) -> __m128d {
    return _mm_cmpneq_pd(d, d);
}

#[inline(always)]
/// Returns flag value is zero
pub unsafe fn _mm_eqzero_pd(d: __m128d) -> __m128d {
    return _mm_cmpeq_pd(d, _mm_set1_pd(0.));
}

#[inline(always)]
/// If mask then `true_vals` otherwise `false_val`
pub unsafe fn _mm_select_pd(mask: __m128d, true_vals: __m128d, false_vals: __m128d) -> __m128d {
    _mm_blendv_pd(false_vals, true_vals, mask)
}

#[inline(always)]
/// Returns flag value is lower than zero
pub unsafe fn _mm_ltzero_pd(d: __m128d) -> __m128d {
    return _mm_cmplt_pd(d, _mm_set1_pd(0.));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_if_inf() {
        unsafe {
            // Test regular
            let value = _mm_set1_pd(23f64);
            let comparison = _mm_isinf_pd(value);
            let flag = _mm_extract_epi64::<0>(_mm_castpd_si128(comparison));
            assert_eq!(flag, 0);
        }

        unsafe {
            // Test negative regular
            let value = _mm_set1_pd(-23f64);
            let comparison = _mm_isinf_pd(value);
            let flag = _mm_extract_epi64::<0>(_mm_castpd_si128(comparison));
            assert_eq!(flag, 0);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_pd(f64::INFINITY);
            let comparison = _mm_isinf_pd(value);
            let flag = _mm_extract_epi64::<0>(_mm_castpd_si128(comparison));
            assert_ne!(flag, 0);
        }
    }

    #[test]
    fn test_absd() {
        unsafe {
            // Test regular
            let value = _mm_set1_pd(23.);
            let comparison = _mm_abs_pd(value);
            let flag = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag, 23.);
        }

        unsafe {
            // Test negative regular
            let value = _mm_set1_pd(-23.);
            let comparison = _mm_abs_pd(value);
            let flag = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag, 23.);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_pd(f64::NEG_INFINITY);
            let comparison = _mm_abs_pd(value);
            let flag = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag.is_infinite(), true);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_pd(f64::INFINITY);
            let comparison = _mm_abs_pd(value);
            let flag = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag.is_infinite(), true);
        }

        unsafe {
            // Test NaN
            let value = _mm_set1_pd(f64::NAN);
            let comparison = _mm_abs_pd(value);
            let flag = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag.is_nan(), true);
        }
    }
}
