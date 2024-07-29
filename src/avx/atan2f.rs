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

use crate::{_mm256_atan_ps, _mm256_eqzero_ps, _mm256_ltzero_ps, _mm256_select_ps};

/// Computes atan for Y,X
#[inline(always)]
pub unsafe fn _mm256_atan2_ps(y: __m256, x: __m256) -> __m256 {
    let zero_x_mask = _mm256_eqzero_ps(x);
    let yx = _mm256_atan_ps(_mm256_div_ps(y, x));
    let mut rad = yx;
    rad = _mm256_select_ps(
        _mm256_and_ps(
            zero_x_mask,
            _mm256_cmp_ps::<_CMP_GE_OS>(y, _mm256_setzero_ps()),
        ),
        _mm256_set1_ps(std::f32::consts::FRAC_PI_2),
        rad,
    ); // x == 0 && y > 0.
    rad = _mm256_select_ps(
        _mm256_and_ps(
            zero_x_mask,
            _mm256_cmp_ps::<_CMP_LE_OS>(y, _mm256_setzero_ps()),
        ),
        _mm256_set1_ps(-std::f32::consts::FRAC_PI_2),
        rad,
    ); // x == 0 && y < 0.
    rad = _mm256_select_ps(
        _mm256_and_ps(
            zero_x_mask,
            _mm256_cmp_ps::<_CMP_LE_OS>(y, _mm256_setzero_ps()),
        ),
        _mm256_set1_ps(0f32),
        rad,
    ); // x == 0 && y == 0.
    let x_lower_than_0 = _mm256_ltzero_ps(x);
    rad = _mm256_select_ps(
        _mm256_and_ps(
            x_lower_than_0,
            _mm256_cmp_ps::<_CMP_GE_OS>(y, _mm256_setzero_ps()),
        ),
        _mm256_add_ps(yx, _mm256_set1_ps(std::f32::consts::PI)),
        rad,
    ); // x < 0 && y >= 0
    rad = _mm256_select_ps(
        _mm256_and_ps(x_lower_than_0, _mm256_ltzero_ps(y)),
        _mm256_add_ps(yx, _mm256_set1_ps(-std::f32::consts::PI)),
        rad,
    ); // x < 0 && y < 0
    rad
}
