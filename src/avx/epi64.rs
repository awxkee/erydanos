/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::shuffle::_mm_shuffle;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
/// Arithmetic shift for i64, shifting with sign bits
pub unsafe fn _mm256_srai_epi64x<const IMM8: i32>(a: __m256i) -> __m256i {
    let m = _mm256_set1_epi64x(1 << (64 - 1));
    let x = _mm256_srli_epi64::<IMM8>(a);
    let result = _mm256_sub_epi64(_mm256_xor_si256(x, m), m); //result = x^m - m
    return result;
}

#[inline(always)]
/// Select true or false values based on masks for i64
pub unsafe fn _mm256_select_epi64(
    mask: __m256i,
    true_vals: __m256i,
    false_vals: __m256i,
) -> __m256i {
    _mm256_castpd_si256(_mm256_blendv_pd(
        _mm256_castsi256_pd(false_vals),
        _mm256_castsi256_pd(true_vals),
        _mm256_castsi256_pd(mask),
    ))
}

#[inline(always)]
/// Takes max for epi64
pub unsafe fn _mm256_max_epi64x(a: __m256i, b: __m256i) -> __m256i {
    let mask = _mm256_cmpgt_epi64(a, b);
    _mm256_blendv_epi8(b, a, mask)
}

#[inline(always)]
/// Takes min for epi64
pub unsafe fn _mm256_min_epi64x(a: __m256i, b: __m256i) -> __m256i {
    let mut mask = _mm256_cmpgt_epi64(a, b);
    let ffs = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
    mask = _mm256_andnot_si256(mask, ffs);
    _mm256_blendv_epi8(b, a, mask)
}

#[inline]
/// Pack 64bytes integers into 32 bytes using unsigned saturation
pub unsafe fn _mm256_packus_epi64(a: __m256i, b: __m256i) -> __m256i {
    let i32_max = _mm256_set1_epi64x(u32::MAX as i64);
    let a = _mm256_select_epi64(_mm256_cmpgt_epi64(a, i32_max), i32_max, a);
    let b = _mm256_select_epi64(_mm256_cmpgt_epi64(a, i32_max), i32_max, b);
    const SHUFFLE_1: i32 = _mm_shuffle(2, 0, 2, 0);
    let combined = _mm256_shuffle_ps::<SHUFFLE_1>(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b));
    const SHUFFLE_2: i32 = _mm_shuffle(3, 1, 2, 0);
    let ordered = _mm256_permute4x64_pd::<SHUFFLE_2>(_mm256_castps_pd(combined));
    return _mm256_castpd_si256(ordered);
}

#[inline]
/// Pack 64bytes integers into 32 bytes using truncation
pub unsafe fn _mm256_packts_epi64(a: __m256i, b: __m256i) -> __m256i {
    const SHUFFLE_1: i32 = _mm_shuffle(2, 0, 2, 0);
    let combined = _mm256_shuffle_ps::<SHUFFLE_1>(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b));
    const SHUFFLE_2: i32 = _mm_shuffle(3, 1, 2, 0);
    let ordered = _mm256_permute4x64_pd::<SHUFFLE_2>(_mm256_castps_pd(combined));
    return _mm256_castpd_si256(ordered);
}

#[inline]
#[allow(dead_code)]
/// Pack 64bytes integers into 32 bytes
pub unsafe fn _mm256_cvtepi64_epi32x(v: __m256i) -> __m128i {
    let vf = _mm256_castsi256_ps(v);
    let hi = _mm256_extractf128_ps::<1>(vf);
    let lo = _mm256_castps256_ps128(vf);
    const FLAGS: i32 = _mm_shuffle(2, 0, 2, 0);
    let packed = _mm_shuffle_ps::<FLAGS>(lo, hi);
    return _mm_castps_si128(packed);
}

#[inline(always)]
/// Multiplies unsigned 64 bytes integers, takes only lower half after multiplication, do not care about overflow
/// Formally it is *_mm256_mullo_epu64*
pub unsafe fn _mm256_mul_epu64(ab: __m256i, cd: __m256i) -> __m256i {
    /* ac = (ab & 0xFFFFFFFF) * (cd & 0xFFFFFFFF); */
    let ac = _mm256_mul_epu32(ab, cd);

    /* b = ab >> 32; */
    let b = _mm256_srli_epi64::<32>(ab);

    /* bc = b * (cd & 0xFFFFFFFF); */
    let bc = _mm256_mul_epu32(b, cd);

    /* d = cd >> 32; */
    let d = _mm256_srli_epi64::<32>(cd);

    /* ad = (ab & 0xFFFFFFFF) * d; */
    let ad = _mm256_mul_epu32(ab, d);

    /* high = bc + ad; */
    let mut high = _mm256_add_epi64(bc, ad);

    /* high <<= 32; */
    high = _mm256_slli_epi64::<32>(high);
    return _mm256_add_epi64(high, ac);
}

#[inline(always)]
/// Multiplies unsigned 64 bytes integers, takes only lower half after multiplication, do not care about overflow
/// Formally it is *_mm_mullo_epi64*
pub unsafe fn _mm256_mul_epi64(ab: __m256i, cd: __m256i) -> __m256i {
    _mm256_mul_epu64(ab, cd)
}

#[inline(always)]
/// Negates i64
pub unsafe fn _mm256_neg_epi64(a: __m256i) -> __m256i {
    let k = _mm256_setzero_si256();
    _mm256_sub_epi64(k, a)
}