/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::{_mm_atan_ps, _mm_eqzero_ps, _mm_ltzero_ps, _mm_select_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes atan for Y,X
#[inline(always)]
pub unsafe fn _mm_atan2_ps(y: __m128, x: __m128) -> __m128 {
    let zero_x_mask = _mm_eqzero_ps(x);
    let yx = _mm_atan_ps(_mm_div_ps(y, x));
    let mut rad = yx;
    rad = _mm_select_ps(
        _mm_and_ps(zero_x_mask, _mm_cmpge_ps(y, _mm_setzero_ps())),
        _mm_set1_ps(std::f32::consts::FRAC_PI_2),
        rad,
    ); // x == 0 && y > 0.
    rad = _mm_select_ps(
        _mm_and_ps(zero_x_mask, _mm_cmple_ps(y, _mm_setzero_ps())),
        _mm_set1_ps(-std::f32::consts::FRAC_PI_2),
        rad,
    ); // x == 0 && y < 0.
    rad = _mm_select_ps(
        _mm_and_ps(zero_x_mask, _mm_cmple_ps(y, _mm_setzero_ps())),
        _mm_set1_ps(0f32),
        rad,
    ); // x == 0 && y == 0.
    let x_lower_than_0 = _mm_ltzero_ps(x);
    rad = _mm_select_ps(
        _mm_and_ps(x_lower_than_0, _mm_cmpge_ps(y, _mm_setzero_ps())),
        _mm_add_ps(yx, _mm_set1_ps(std::f32::consts::PI)),
        rad,
    ); // x < 0 && y >= 0
    rad = _mm_select_ps(
        _mm_and_ps(x_lower_than_0, _mm_ltzero_ps(y)),
        _mm_add_ps(yx, _mm_set1_ps(-std::f32::consts::PI)),
        rad,
    ); // x < 0 && y < 0
    rad
}
