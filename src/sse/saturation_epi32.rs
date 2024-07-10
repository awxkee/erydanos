/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::sse::epi32::_mm_blendv_epi32;
use crate::sse::unsigned::_mm_cmplt_epu32;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
/// Add unsigned 32 bytes integers using saturation
pub unsafe fn _mm_adds_epu32(a: __m128i, b: __m128i) -> __m128i {
    let result = _mm_add_epi32(a, b);
    let mask = _mm_cmplt_epu32(result, a);
    #[allow(overflowing_literals)]
    let saturated = _mm_set1_epi32(0xFFFFFFFF);
    _mm_or_si128(
        _mm_and_si128(mask, saturated),
        _mm_andnot_si128(mask, result),
    )
}

#[inline(always)]
/// Add signed 32 bytes integers using saturation
pub unsafe fn _mm_adds_epi32(a: __m128i, b: __m128i) -> __m128i {
    let res = _mm_add_epi32(a, b);

    _mm_blendv_epi32(
        res,
        _mm_blendv_epi32(_mm_set1_epi32(i32::MIN), _mm_set1_epi32(i32::MAX), res),
        _mm_xor_si128(b, _mm_cmpgt_epi32(a, res)),
    )
}

#[inline(always)]
/// Subtract signed integers 32 using saturation
pub unsafe fn _mm_subs_epi32(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_sub_epi32(lhs, rhs);

    _mm_blendv_epi32(
        res,
        _mm_blendv_epi32(_mm_set1_epi32(i32::MIN), _mm_set1_epi32(i32::MAX), res),
        _mm_xor_si128(
            _mm_cmpgt_epi32(rhs, _mm_setzero_si128()),
            _mm_cmpgt_epi32(lhs, res),
        ),
    )
}
