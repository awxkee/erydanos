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
pub unsafe fn _mm_blendv_epi32(xmm0: __m128i, xmm1: __m128i, mask: __m128i) -> __m128i {
    _mm_castps_si128(_mm_blendv_ps(
        _mm_castsi128_ps(xmm0),
        _mm_castsi128_ps(xmm1),
        _mm_castsi128_ps(mask),
    ))
}

#[inline(always)]
/// If mask then `true_vals` otherwise `false_val`
pub unsafe fn _mm_select_epi32(mask: __m128i, true_vals: __m128i, false_vals: __m128i) -> __m128i {
    _mm_blendv_epi32(false_vals, true_vals, mask)
}

#[inline(always)]
/// Negates signed 32 bytes integer
pub unsafe fn _mm_neg_epi32(d: __m128i) -> __m128i {
    _mm_sub_epi32(_mm_set1_epi32(0), d)
}
