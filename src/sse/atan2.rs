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

use crate::{_mm_atan_pd, _mm_eqzero_pd, _mm_select_pd};
use crate::sse::general::_mm_ltzero_pd;

/// Computes atan for Y,X
#[inline(always)]
pub unsafe fn _mm_atan2_pd(y: __m128d, x: __m128d) -> __m128d {
    let zero_x_mask = _mm_eqzero_pd(x);
    let yx = _mm_atan_pd(_mm_div_pd(y, x));
    let mut rad = yx;
    rad = _mm_select_pd(
        _mm_and_pd(zero_x_mask, _mm_cmpge_pd(y, _mm_setzero_pd())),
        _mm_set1_pd(std::f64::consts::FRAC_PI_2),
        rad,
    ); // x == 0 && y > 0.
    rad = _mm_select_pd(
        _mm_and_pd(zero_x_mask, _mm_cmple_pd(y, _mm_setzero_pd())),
        _mm_set1_pd(-std::f64::consts::FRAC_PI_2),
        rad,
    ); // x == 0 && y < 0.
    rad = _mm_select_pd(
        _mm_and_pd(zero_x_mask, _mm_cmple_pd(y, _mm_setzero_pd())),
        _mm_set1_pd(0.),
        rad,
    ); // x == 0 && y == 0.
    let x_lower_than_0 = _mm_ltzero_pd(x);
    rad = _mm_select_pd(
        _mm_and_pd(x_lower_than_0, _mm_cmpge_pd(y, _mm_setzero_pd())),
        _mm_add_pd(yx, _mm_set1_pd(std::f64::consts::PI)),
        rad,
    ); // x < 0 && y >= 0
    rad = _mm_select_pd(
        _mm_and_pd(x_lower_than_0, _mm_ltzero_pd(y)),
        _mm_add_pd(yx, _mm_set1_pd(-std::f64::consts::PI)),
        rad,
    ); // x < 0 && y < 0
    rad
}
