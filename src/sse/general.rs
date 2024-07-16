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

use crate::_mm_shuffle;
use crate::sse::epi64::{_mm_sllv_epi64x, _mm_srlv_epi64x};

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
/// If mask then `true_vals` otherwise `false_val`
pub unsafe fn _mm_selecti_pd(mask: __m128i, true_vals: __m128d, false_vals: __m128d) -> __m128d {
    _mm_blendv_pd(false_vals, true_vals, _mm_castsi128_pd(mask))
}

#[inline(always)]
/// Returns flag value is lower than zero
pub unsafe fn _mm_ltzero_pd(d: __m128d) -> __m128d {
    return _mm_cmplt_pd(d, _mm_set1_pd(0.));
}

#[inline(always)]
/// Computes 2^n in f64 form for signed 64 bits integers, returns f64 in bits
pub unsafe fn _mm_pow2i_epi64(n: __m128i) -> __m128i {
    let j = _mm_slli_epi64::<52>(_mm_add_epi64(n, _mm_set1_epi32(0x3ff)));
    j
}

#[inline(always)]
/// Converts double into signed 64 bytes int with truncation
pub unsafe fn _mm_cvtpd_epi64(v: __m128d) -> __m128i {
    let k51 = _mm_set1_epi64x(51 + 0x3FF);

    // Exponent indicates whether the number can be represented as uint64_t.
    let biased_exp = _mm_and_si128(
        _mm_srli_epi64::<52>(_mm_castpd_si128(v)),
        _mm_set1_epi64x(0x7FF),
    );
    let mantissa = _mm_and_si128(_mm_castpd_si128(v), _mm_set1_epi64x((1 << 52) - 1));

    // Calculate left and right shifts to move mantissa into place.
    let shift_right = _mm_subs_epu16(k51, biased_exp);
    let shift_left = _mm_subs_epu16(biased_exp, k51);

    // Shift mantissa into place.
    let shifted = _mm_srli_epi64::<1>(_mm_srlv_epi64x(
        _mm_sllv_epi64x(mantissa, shift_left),
        shift_right,
    ));
    // Include implicit 1-bit.
    let implicit_bit_shifted = _mm_srlv_epi64x(
        _mm_sllv_epi64x(_mm_set1_epi64x(1 << 51), shift_left),
        shift_right,
    );
    let magnitude = _mm_or_si128(shifted, implicit_bit_shifted);

    // Fill each 64-bit part with sign bits.
    const SIGN_MASK: i32 = _mm_shuffle(3, 3, 1, 1);
    let sign_mask = _mm_shuffle_epi32::<SIGN_MASK>(_mm_srai_epi32::<31>(_mm_castpd_si128(v)));
    // Adjust for negative values.
    let sign_adjusted = _mm_sub_epi64(_mm_xor_si128(magnitude, sign_mask), sign_mask);

    // 0xFF is exp < 64
    const UPPER_BOUND_MASK: i32 = _mm_shuffle(2, 2, 0, 0);
    let upper_bound_mask = _mm_shuffle_epi32::<UPPER_BOUND_MASK>(_mm_cmpgt_epi32(
        _mm_set1_epi32(64 + 0x3FF),
        biased_exp,
    ));
    // Saturate overflow values to INT64_MIN.
    let bounded = _mm_or_si128(
        _mm_and_si128(upper_bound_mask, sign_adjusted),
        _mm_andnot_si128(upper_bound_mask, _mm_set1_epi64x(i64::MAX)),
    );

    return bounded;
}

#[inline(always)]
/// Converts double into unsigned int 64 bytes with truncation
pub unsafe fn _mm_cvtpd_epu64(v: __m128d) -> __m128i {
    let k51 = _mm_set1_epi64x(51 + 0x3FF);

    // Exponent indicates whether the number can be represented as uint64_t.
    let biased_exp = _mm_and_si128(
        _mm_srli_epi64::<52>(_mm_castpd_si128(v)),
        _mm_set1_epi64x(0x7FF),
    );
    let mantissa = _mm_and_si128(_mm_castpd_si128(v), _mm_set1_epi64x((1 << 52) - 1));

    // Calculate left and right shifts to move mantissa into place.
    let shift_right = _mm_subs_epu16(k51, biased_exp);
    let shift_left = _mm_subs_epu16(biased_exp, k51);

    // Shift mantissa into place.
    let shifted = _mm_srli_epi64::<1>(_mm_srlv_epi64x(
        _mm_sllv_epi64x(mantissa, shift_left),
        shift_right,
    ));
    // Include implicit 1-bit.
    let implicit_bit_shifted = _mm_srlv_epi64x(
        _mm_sllv_epi64x(_mm_set1_epi64x(1 << 51), shift_left),
        shift_right,
    );
    let magnitude = _mm_or_si128(shifted, implicit_bit_shifted);

    // Fill each 64-bit part with sign bits.
    const SIGN_MASK_SHUFFLE: i32 = _mm_shuffle(3, 3, 1, 1);
    let sign_mask =
        _mm_shuffle_epi32::<SIGN_MASK_SHUFFLE>(_mm_srai_epi32::<31>(_mm_castpd_si128(v)));
    // Mask out negative values to 0.
    let lower_bounded = _mm_andnot_si128(sign_mask, magnitude);

    // 0xFF is exp < 64
    const SIGN_UPPER_BOUND_SHUFFLE: i32 = _mm_shuffle(2, 2, 0, 0);
    let upper_bound_mask = _mm_shuffle_epi32::<SIGN_UPPER_BOUND_SHUFFLE>(_mm_cmpgt_epi32(
        _mm_set1_epi32(64 + 0x3FF),
        biased_exp,
    ));
    // Mask out overflow values to 0.
    let fully_bounded = _mm_and_si128(lower_bounded, upper_bound_mask);

    return fully_bounded;
}

#[inline(always)]
/// Rounds and takes integral part 64 bytes from double
pub unsafe fn _mm_rint_pd(f: __m128d) -> __m128i {
    const ROUNDING_FLAG: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    let k = _mm_round_pd::<ROUNDING_FLAG>(f);
    _mm_cvtpd_epi64(k)
}

#[inline(always)]
/// Copies sign from `y` to `x`
pub unsafe fn _mm_copysign_pd(x: __m128d, y: __m128d) -> __m128d {
    _mm_castsi128_pd(_mm_xor_si128(
        _mm_andnot_si128(_mm_castpd_si128(_mm_set1_pd(-0.0f64)), _mm_castpd_si128(x)),
        _mm_and_si128(_mm_castpd_si128(_mm_set1_pd(-0.0f64)), _mm_castpd_si128(y)),
    ))
}

#[inline(always)]
/// Returns flag value is Neg Infinity
pub unsafe fn _mm_isneginf_pd(d: __m128d) -> __m128d {
    return _mm_cmpeq_pd(d, _mm_set1_pd(f64::NEG_INFINITY));
}

#[inline(always)]
/// Checks if arguments is integral value
pub unsafe fn _mm_isintegral_pd(d: __m128d) -> __m128d {
    return _mm_cmpeq_pd(d, _mm_floor_pd(d));
}

#[inline(always)]
/// Checks if arguments is not integral value
pub unsafe fn _mm_isnotintegral_pd(d: __m128d) -> __m128d {
    return _mm_cmpneq_pd(d, _mm_floor_pd(d));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvtpd_epi64() {
        unsafe {
            // Test regular
            let value = _mm_set1_pd(23f64);
            let comparison = _mm_cvtpd_epi64(value);
            let flag = _mm_extract_epi64::<0>(comparison);
            assert_eq!(flag, 23);
        }

        unsafe {
            // Test negative regular
            let value = _mm_set1_pd(-23f64);
            let comparison = _mm_cvtpd_epi64(value);
            let flag = _mm_extract_epi64::<0>(comparison);
            assert_eq!(flag, -23);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_pd(f64::INFINITY);
            let comparison = _mm_cvtpd_epu64(value);
            let flag = _mm_extract_epi64::<0>(comparison);
            assert_eq!(flag, 0);
        }
    }

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

    #[test]
    fn test_copysignd() {
        unsafe {
            let value = _mm_set1_pd(23f64);
            let other = _mm_set1_pd(-2f64);
            let comparison = _mm_copysign_pd(value, other);
            let flag = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag, -23f64);
        }

        unsafe {
            let value = _mm_set1_pd(23f64);
            let other = _mm_set1_pd(2f64);
            let comparison = _mm_copysign_pd(value, other);
            let flag = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag, 23f64);
        }
    }
}
