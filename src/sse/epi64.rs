/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::_mm_cmplt_epi64;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
/// Mod function for i64
pub unsafe fn _mm_abs_epi64(a: __m128i) -> __m128i {
    _mm_select_epi64(_mm_cmplt_epi64(a, _mm_setzero_si128()), _mm_neg_epi64(a), a)
}

#[inline(always)]
/// Negates i64
pub unsafe fn _mm_neg_epi64(a: __m128i) -> __m128i {
    let k = _mm_setzero_si128();
    _mm_sub_epi64(k, a)
}

#[inline(always)]
/// Select true or false values based on masks for i64
pub unsafe fn _mm_select_epi64(mask: __m128i, true_vals: __m128i, false_vals: __m128i) -> __m128i {
    _mm_castpd_si128(_mm_blendv_pd(
        _mm_castsi128_pd(false_vals),
        _mm_castsi128_pd(true_vals),
        _mm_castsi128_pd(mask),
    ))
}

#[inline(always)]
/// Multiplies unsigned 64 bytes integers
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

#[inline(never)]
/// Multiplies signed 64 bytes integers
pub unsafe fn _mm_mul_epi64(ab: __m128i, cd: __m128i) -> __m128i {
    let sign_ab = _mm_srli_epi64::<63>(ab);
    let sign_cd = _mm_srli_epi64::<63>(cd);
    let sign = _mm_xor_si128(sign_ab, sign_cd);
    let uab = _mm_abs_epi64(ab);
    let ucd = _mm_abs_epi64(cd);
    let product = _mm_mul_epu64(uab, ucd);
    let o = _mm_select_epi64(
        _mm_cmpeq_epi64(sign, _mm_setzero_si128()),
        product,
        _mm_neg_epi64(product),
    );
    return o;
}

#[inline(always)]
pub unsafe fn _mm_blendv_epi64(xmm0: __m128i, xmm1: __m128i, mask: __m128i) -> __m128i {
    _mm_castpd_si128(_mm_blendv_pd(
        _mm_castsi128_pd(xmm0),
        _mm_castsi128_pd(xmm1),
        _mm_castsi128_pd(mask),
    ))
}

#[inline(always)]
pub unsafe fn _mm_setr_epi64x(a: i64, b: i64) -> __m128i {
    _mm_set_epi64x(b, a)
}

#[inline(always)]
#[rustfmt::skip]
/// Converts signed 64 bit integers into double
pub unsafe fn _mm_cvtepi64_pd(v: __m128i) -> __m128d {
    let magic_i_lo   = _mm_set1_epi64x(0x4330000000000000); // 2^52               encoded as floating-point
    let magic_i_hi32 = _mm_set1_epi64x(0x4530000080000000); // 2^84 + 2^63        encoded as floating-point
    let magic_i_all  = _mm_set1_epi64x(0x4530000080100000); // 2^84 + 2^63 + 2^52 encoded as floating-point
    let magic_d_all  = _mm_castsi128_pd(magic_i_all);

    let     v_lo     = _mm_blend_epi16::<0b00110011>(magic_i_lo, v);      // Blend the 32 lowest significant bits of v with magic_int_lo
    let mut v_hi     = _mm_srli_epi64::<32>(v);                           // Extract the 32 most significant bits of v
    v_hi     = _mm_xor_si128(v_hi, magic_i_hi32);               // Flip the msb of v_hi and blend with 0x45300000
    let     v_hi_dbl = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all); // Compute in double precision:
    _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

#[inline(always)]
/// Shifts packed 64-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros,
pub unsafe fn _mm_srlv_epi64x(a: __m128i, count: __m128i) -> __m128i {
    let shift_low = _mm_srl_epi64(a, count); // high 64 is garbage
    let count_high = _mm_unpackhi_epi64(count, count); // broadcast the high element
    let shift_high = _mm_srl_epi64(a, count_high); // low 64 is garbage

    // use movsd as a blend.
    return _mm_castpd_si128(_mm_move_sd(
        _mm_castsi128_pd(shift_high),
        _mm_castsi128_pd(shift_low),
    ));
}

#[inline(always)]
/// Shifts packed 64-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and returns the result.
pub unsafe fn _mm_sllv_epi64x(a: __m128i, count: __m128i) -> __m128i {
    let shift_low = _mm_sll_epi64(a, count); // high 64 is garbage
    let count_high = _mm_unpackhi_epi64(count, count); // broadcast the high element
    let shift_high = _mm_sll_epi64(a, count_high); // low 64 is garbage

    // use movsd as a blend.
    return _mm_castpd_si128(_mm_move_sd(
        _mm_castsi128_pd(shift_high),
        _mm_castsi128_pd(shift_low),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_mm_extract_pd;

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
