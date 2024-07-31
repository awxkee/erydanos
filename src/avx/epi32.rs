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
/// Negates signed 32 bytes integer
pub unsafe fn _mm256_neg_epi32(d: __m256i) -> __m256i {
    _mm256_sub_epi32(_mm256_setzero_si256(), d)
}
