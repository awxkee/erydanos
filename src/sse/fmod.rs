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
pub unsafe fn _mm_fmod_ps(a: __m128, b: __m128) -> __m128 {
    let dividend_vec = a;
    let divisor_vec = b;
    let division = _mm_mul_ps(dividend_vec, _mm_rcp_ps(divisor_vec)); // Perform division
    let int_part = _mm_floor_ps(division); // Get the integer part using floor
    let product = _mm_mul_ps(int_part, divisor_vec); // Multiply the integer part by the divisor
    let remainder = _mm_sub_ps(dividend_vec, product); // Subtract the product from the dividend
    remainder
}
