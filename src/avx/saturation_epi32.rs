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

use crate::{_mm256_blendv_epi32, _mm256_cmplt_epu32};

#[inline(always)]
/// Add unsigned 32 bytes integers using saturation
pub unsafe fn _mm256_adds_epu32(a: __m256i, b: __m256i) -> __m256i {
    let result = _mm256_add_epi32(a, b);
    let mask = _mm256_cmplt_epu32(result, a);
    #[allow(overflowing_literals)]
    let saturated = _mm256_set1_epi32(0xFFFFFFFF);
    _mm256_or_si256(
        _mm256_and_si256(mask, saturated),
        _mm256_andnot_si256(mask, result),
    )
}

#[inline(always)]
/// Add signed 32 bytes integers using saturation
pub unsafe fn _mm256_adds_epi32(a: __m256i, b: __m256i) -> __m256i {
    let res = _mm256_add_epi32(a, b);

    _mm256_blendv_epi32(
        res,
        _mm256_blendv_epi32(
            _mm256_set1_epi32(i32::MIN),
            _mm256_set1_epi32(i32::MAX),
            res,
        ),
        _mm256_xor_si256(b, _mm256_cmpgt_epi32(a, res)),
    )
}

#[inline(always)]
/// Subtract signed integers 32 using saturation
pub unsafe fn _mm256_subs_epi32(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_sub_epi32(lhs, rhs);

    _mm256_blendv_epi32(
        res,
        _mm256_blendv_epi32(
            _mm256_set1_epi32(i32::MIN),
            _mm256_set1_epi32(i32::MAX),
            res,
        ),
        _mm256_xor_si256(
            _mm256_cmpgt_epi32(rhs, _mm256_setzero_si256()),
            _mm256_cmpgt_epi32(lhs, res),
        ),
    )
}
