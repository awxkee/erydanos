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

#[inline]
/// Negates signed 32 bytes integer
pub unsafe fn _mm256_neg_epi32(d: __m256i) -> __m256i {
    _mm256_sub_epi32(_mm256_setzero_si256(), d)
}

#[inline]
/// Compare *greater than or equal to* unsigned 16,
pub unsafe fn _mm256_cmpge_epu16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpeq_epi16(_mm256_max_epu16(a, b), a)
}

#[inline]
/// Compare *less than or equal to* unsigned 16,
pub unsafe fn _mm256_cmple_epu16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpge_epu16(b, a)
}

#[inline]
/// Compare *greater than* unsigned 16,
pub unsafe fn _mm256_cmpgt_epu16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(_mm256_cmple_epu16(a, b), _mm256_set1_epi16(-1))
}

#[inline]
/// Compare *less than* unsigned 16,
pub unsafe fn _mm256_cmplt_epu16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epu16(b, a)
}

#[inline]
/// Compare *greater than or equal to* unsigned 32,
pub unsafe fn _mm256_cmpge_epu32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpeq_epi32(_mm256_max_epu32(a, b), a)
}

#[inline]
/// Compare *less than or equal to* unsigned 32,
pub unsafe fn _mm256_cmple_epu32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpge_epu16(b, a)
}

#[inline]
/// Compare *greater than* unsigned 32,
pub unsafe fn _mm256_cmpgt_epu32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(_mm256_cmple_epu32(a, b), _mm256_set1_epi32(-1))
}

#[inline]
/// Compare *less than* unsigned 16,
pub unsafe fn _mm256_cmplt_epu32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epu32(b, a)
}

#[inline]
pub unsafe fn _mm256_blendv_epi32(xmm0: __m256i, xmm1: __m256i, mask: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(xmm0),
        _mm256_castsi256_ps(xmm1),
        _mm256_castsi256_ps(mask),
    ))
}
