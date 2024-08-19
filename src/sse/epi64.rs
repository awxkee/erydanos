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

use crate::shuffle::_mm_shuffle;
use crate::{_mm_cmplt_epi64, _mm_max_epi64x, _mm_min_epi64x};

#[inline]
#[target_feature(enable = "sse4.1")]
/// Mod function for i64
pub unsafe fn _mm_abs_epi64(a: __m128i) -> __m128i {
    _mm_select_epi64(_mm_cmplt_epi64(a, _mm_setzero_si128()), _mm_neg_epi64(a), a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Negates i64
pub unsafe fn _mm_neg_epi64(a: __m128i) -> __m128i {
    let k = _mm_setzero_si128();
    _mm_sub_epi64(k, a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Select true or false values based on masks for i64
pub unsafe fn _mm_select_epi64(mask: __m128i, true_vals: __m128i, false_vals: __m128i) -> __m128i {
    _mm_castpd_si128(_mm_blendv_pd(
        _mm_castsi128_pd(false_vals),
        _mm_castsi128_pd(true_vals),
        _mm_castsi128_pd(mask),
    ))
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Multiplies unsigned 64 bytes integers, takes only lower half after multiplication, do not care about overflow
/// Formally it is *_mm_mullo_epu64*
pub unsafe fn _mm_mul_epu64(ab: __m128i, cd: __m128i) -> __m128i {
    /* ac = (ab & 0xFFFFFFFF) * (cd & 0xFFFFFFFF); */
    let ac = _mm_mul_epu32(ab, cd);

    /* b = ab >> 32; */
    let b = _mm_srli_epi64::<32>(ab);

    /* bc = b * (cd & 0xFFFFFFFF); */
    let bc = _mm_mul_epu32(b, cd);

    /* d = cd >> 32; */
    let d = _mm_srli_epi64::<32>(cd);

    /* ad = (ab & 0xFFFFFFFF) * d; */
    let ad = _mm_mul_epu32(ab, d);

    /* high = bc + ad; */
    let mut high = _mm_add_epi64(bc, ad);

    /* high <<= 32; */
    high = _mm_slli_epi64::<32>(high);
    return _mm_add_epi64(high, ac);
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Multiplies unsigned 64 bytes integers, takes only lower half after multiplication, do not care about overflow
/// Formally it is *_mm_mullo_epi64*
pub unsafe fn _mm_mul_epi64(ab: __m128i, cd: __m128i) -> __m128i {
    _mm_mul_epu64(ab, cd)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_blendv_epi64(xmm0: __m128i, xmm1: __m128i, mask: __m128i) -> __m128i {
    _mm_castpd_si128(_mm_blendv_pd(
        _mm_castsi128_pd(xmm0),
        _mm_castsi128_pd(xmm1),
        _mm_castsi128_pd(mask),
    ))
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_setr_epi64x(a: i64, b: i64) -> __m128i {
    _mm_set_epi64x(b, a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[rustfmt::skip]
/// Converts signed 64-bit integers into double
pub unsafe fn _mm_cvtepi64_pd(v: __m128i) -> __m128d {
    let magic_i_lo = _mm_set1_epi64x(0x4330000000000000); // 2^52               encoded as floating-point
    let magic_i_hi32 = _mm_set1_epi64x(0x4530000080000000); // 2^84 + 2^63        encoded as floating-point
    let magic_i_all = _mm_set1_epi64x(0x4530000080100000); // 2^84 + 2^63 + 2^52 encoded as floating-point
    let magic_d_all = _mm_castsi128_pd(magic_i_all);

    let v_lo = _mm_blend_epi16::<0b00110011>(magic_i_lo, v);      // Blend the 32 lowest significant bits of v with magic_int_lo
    let mut v_hi = _mm_srli_epi64::<32>(v);                           // Extract the 32 most significant bits of v
    v_hi = _mm_xor_si128(v_hi, magic_i_hi32);               // Flip the msb of v_hi and blend with 0x45300000
    let v_hi_dbl = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all); // Compute in double precision:
    _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Shifts packed 64-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros,
pub unsafe fn _mm_srlv_epi64x(a: __m128i, count: __m128i) -> __m128i {
    let shift_low = _mm_srl_epi64(a, count); // high 64 is garbage
    let count_high = _mm_unpackhi_epi64(count, count); // broadcast the high element
    let shift_high = _mm_srl_epi64(a, count_high); // low 64 is garbage

    // use movsd as a blend.
    _mm_castpd_si128(_mm_move_sd(
        _mm_castsi128_pd(shift_high),
        _mm_castsi128_pd(shift_low),
    ))
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Shifts packed 64-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and returns the result.
pub unsafe fn _mm_sllv_epi64x(a: __m128i, count: __m128i) -> __m128i {
    let shift_low = _mm_sll_epi64(a, count); // high 64 is garbage
    let count_high = _mm_unpackhi_epi64(count, count); // broadcast the high element
    let shift_high = _mm_sll_epi64(a, count_high); // low 64 is garbage

    // use movsd as a blend.
    _mm_castpd_si128(_mm_move_sd(
        _mm_castsi128_pd(shift_high),
        _mm_castsi128_pd(shift_low),
    ))
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Extracts i64 value
pub unsafe fn _mm_extract_epi64x<const IMM: i32>(d: __m128i) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        return if IMM == 0 {
            _mm_cvtsi128_si64(d)
        } else {
            _mm_extract_epi64::<IMM>(d)
        };
    }
    #[cfg(target_arch = "x86")]
    {
        let (low, high);
        if IMM == 0 {
            low = _mm_cvtsi128_si32(d);
            high = _mm_cvtsi128_si32(_mm_srli_si128::<4>(d));
        } else {
            low = _mm_cvtsi128_si32(_mm_srli_si128::<8>(d));
            high = _mm_cvtsi128_si32(_mm_srli_si128::<12>(d));
        }
        return ((high as i64) << 32) | low as i64;
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Bitwise not epi64
pub unsafe fn _mm_not_epi64(a: __m128i) -> __m128i {
    #[allow(overflowing_literals)]
    let all_ones = _mm_set1_epi64x(0xffff_ffff_ffff_ffff);
    return _mm_xor_si128(a, all_ones);
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Arithmetic shift for i64, shifting with sign bits
pub unsafe fn _mm_srai_epi64x<const IMM8: i32>(a: __m128i) -> __m128i {
    let m = _mm_set1_epi64x(1 << (64 - 1));
    let x = _mm_srli_epi64::<IMM8>(a);
    let result = _mm_sub_epi64(_mm_xor_si128(x, m), m); //result = x^m - m
    return result;
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Packs integers 64 bits use unsigned saturation
pub unsafe fn _mm_packus_epi64(a: __m128i, b: __m128i) -> __m128i {
    let i32_max = _mm_set1_epi64x(u32::MAX as i64);
    let a = _mm_select_epi64(_mm_cmpgt_epi64(a, i32_max), i32_max, a);
    let b = _mm_select_epi64(_mm_cmpgt_epi64(b, i32_max), i32_max, b);
    const SHUFFLE_MASK: i32 = _mm_shuffle(3, 1, 2, 0);
    let a = _mm_shuffle_epi32::<SHUFFLE_MASK>(a);
    let b1 = _mm_shuffle_epi32::<SHUFFLE_MASK>(b);
    let moved = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b1)));
    moved
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Packs integers 64 bits use signed saturation
pub unsafe fn _mm_packs_epi64(a: __m128i, b: __m128i) -> __m128i {
    let i32_max = _mm_set1_epi64x(i32::MAX as i64);
    let i32_min = _mm_set1_epi64x(i32::MIN as i64);
    let a = _mm_max_epi64x(_mm_min_epi64x(a, i32_max), i32_min);
    let b = _mm_max_epi64x(_mm_min_epi64x(b, i32_max), i32_min);
    const SHUFFLE_MASK: i32 = _mm_shuffle(3, 1, 2, 0);
    let a = _mm_shuffle_epi32::<SHUFFLE_MASK>(a);
    let b1 = _mm_shuffle_epi32::<SHUFFLE_MASK>(b);
    let moved = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b1)));
    moved
}

#[inline]
#[target_feature(enable = "sse4.1")]
/// Packs integers 64 bits use truncating, only lower half of i64 will be used
pub unsafe fn _mm_packts_epi64(a: __m128i, b: __m128i) -> __m128i {
    const SHUFFLE_MASK: i32 = _mm_shuffle(3, 1, 2, 0);
    let a = _mm_shuffle_epi32::<SHUFFLE_MASK>(a);
    let b1 = _mm_shuffle_epi32::<SHUFFLE_MASK>(b);
    let moved = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b1)));
    moved
}

#[cfg(test)]
mod tests {
    use crate::_mm_extract_pd;

    use super::*;

    #[test]
    fn test_mul_lo_epi64() {
        unsafe {
            // Test regular
            let value1 = _mm_set1_epi64x(24);
            let value2 = _mm_set1_epi64x(2);
            let product = _mm_mul_epu64(value1, value2);
            let flag = _mm_extract_epi64x::<0>(product);
            assert_eq!(flag, 24 * 2);
        }

        unsafe {
            // Test regular
            let value1 = _mm_set1_epi64x(-27);
            let value2 = _mm_set1_epi64x(2);
            let product = _mm_mul_epi64(value1, value2);
            let flag = _mm_extract_epi64x::<0>(product);
            assert_eq!(flag, -27 * 2);
        }
        unsafe {
            // Test regular
            let value1 = _mm_set1_epi64x(27);
            let value2 = _mm_set1_epi64x(-2);
            let product = _mm_mul_epi64(value1, value2);
            let flag = _mm_extract_epi64x::<0>(product);
            assert_eq!(flag, -27 * 2);
        }
        unsafe {
            // Test regular
            let value1 = _mm_set1_epi64x(-27);
            let value2 = _mm_set1_epi64x(-2);
            let product = _mm_mul_epi64(value1, value2);
            let flag = _mm_extract_epi64x::<0>(product);
            assert_eq!(flag, 27 * 2);
        }
        unsafe {
            // Test regular
            let value1 = _mm_set1_epi64x(i32::MAX as i64);
            let value2 = _mm_set1_epi64x(2);
            let product = _mm_mul_epi64(value1, value2);
            let flag = _mm_extract_epi64x::<0>(product);
            assert_eq!(flag, i32::MAX as i64 * 2);
        }
    }

    #[test]
    fn test_cvtepi64_pd() {
        unsafe {
            // Test regular
            let value = _mm_set1_epi64x(24);
            let converted = _mm_cvtepi64_pd(value);
            let flag = _mm_extract_pd::<0>(converted);
            assert_eq!(flag, 24.);
        }
    }
}
