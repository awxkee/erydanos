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
/// Computes fmod for f64
pub unsafe fn _mm_fmod_pd(a: __m128d, b: __m128d) -> __m128d {
    let dividend_vec = a;
    let divisor_vec = b;
    let division = _mm_mul_pd(dividend_vec, _mm_div_pd(_mm_set1_pd(1.), divisor_vec));
    let int_part = _mm_floor_pd(division);
    let product = _mm_mul_pd(int_part, divisor_vec);
    let remainder = _mm_sub_pd(dividend_vec, product);
    remainder
}
