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

use crate::{_mm256_atan_pd, _mm256_select_pd};

/// Computes atan for Y,X
#[inline]
pub unsafe fn _mm256_atan2_pd(y: __m256d, x: __m256d) -> __m256d {
    let zero_x_mask = _mm256_cmp_pd::<_CMP_EQ_OS>(x, _mm256_setzero_pd());
    let yx = _mm256_atan_pd(_mm256_div_pd(y, x));
    let mut rad = yx;
    rad = _mm256_select_pd(
        _mm256_and_pd(
            zero_x_mask,
            _mm256_cmp_pd::<_CMP_GE_OS>(y, _mm256_setzero_pd()),
        ),
        _mm256_set1_pd(std::f64::consts::FRAC_PI_2),
        rad,
    ); // x == 0 && y > 0.
    rad = _mm256_select_pd(
        _mm256_and_pd(
            zero_x_mask,
            _mm256_cmp_pd::<_CMP_LE_OS>(y, _mm256_setzero_pd()),
        ),
        _mm256_set1_pd(-std::f64::consts::FRAC_PI_2),
        rad,
    ); // x == 0 && y < 0.
    rad = _mm256_select_pd(
        _mm256_and_pd(
            zero_x_mask,
            _mm256_cmp_pd::<_CMP_LE_OS>(y, _mm256_setzero_pd()),
        ),
        _mm256_set1_pd(0.),
        rad,
    ); // x == 0 && y == 0.
    let x_lower_than_0 = _mm256_cmp_pd::<_CMP_LT_OS>(x, _mm256_setzero_pd());
    rad = _mm256_select_pd(
        _mm256_and_pd(
            x_lower_than_0,
            _mm256_cmp_pd::<_CMP_GE_OS>(y, _mm256_setzero_pd()),
        ),
        _mm256_add_pd(yx, _mm256_set1_pd(std::f64::consts::PI)),
        rad,
    ); // x < 0 && y >= 0
    rad = _mm256_select_pd(
        _mm256_and_pd(
            x_lower_than_0,
            _mm256_cmp_pd::<_CMP_LT_OS>(y, _mm256_setzero_pd()),
        ),
        _mm256_add_pd(yx, _mm256_set1_pd(-std::f64::consts::PI)),
        rad,
    ); // x < 0 && y < 0
    rad
}
