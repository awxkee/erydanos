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
/// Mod function for i64
pub unsafe fn _mm_abs_epi64(a: __m128i) -> __m128i {
    #[allow(overflowing_literals)]
    let sign_mask = _mm_set1_epi64x(0x8000000000000000i64);
    let sign_bits = _mm_and_si128(a, sign_mask);
    let negated = _mm_sub_epi64(_mm_xor_si128(a, sign_bits), sign_bits);
    return negated;
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

#[inline(always)]
/// Multiplies signed 64 bytes integers
pub unsafe fn _mm_mul_epi64(ab: __m128i, cd: __m128i) -> __m128i {
    let sign_ab = _mm_srli_epi64::<63>(ab);
    let sign_cd = _mm_srli_epi64::<63>(cd);
    let sign = _mm_xor_si128(sign_ab, sign_cd);
    let uab = _mm_abs_epi64(ab);
    let ucd = _mm_abs_epi64(cd);
    let product = _mm_mul_epu64(uab, ucd);
    _mm_select_epi64(
        _mm_cmpeq_epi64(sign, _mm_setzero_si128()),
        product,
        _mm_neg_epi64(product),
    )
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