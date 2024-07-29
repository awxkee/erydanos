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
/// Computes fmod for f32
pub unsafe fn _mm256_fmod_ps(a: __m256, b: __m256) -> __m256 {
    let dividend_vec = a;
    let divisor_vec = b;
    let division = _mm256_mul_ps(dividend_vec, _mm256_div_ps(_mm256_set1_ps(1.), divisor_vec));
    let int_part = _mm256_floor_ps(division);
    let product = _mm256_mul_ps(int_part, divisor_vec);
    let remainder = _mm256_sub_ps(dividend_vec, product);
    remainder
}
