/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::atanf::{
    ATAN_POLY_1_F, ATAN_POLY_2_F, ATAN_POLY_3_F, ATAN_POLY_4_F, ATAN_POLY_5_F, ATAN_POLY_6_F,
    ATAN_POLY_7_F, ATAN_POLY_8_F, ATAN_POLY_9_F,
};
use crate::{_mm_abs_ps, _mm_ltzero_ps, _mm_mlaf_ps, _mm_neg_ps, _mm_select_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes Atan function with *ULP 1.0* error
#[inline(always)]
pub unsafe fn _mm_atan_ps(x: __m128) -> __m128 {
    let negative_mask = _mm_ltzero_ps(x);
    let d = _mm_abs_ps(x);
    let more_than_one_mask = _mm_cmpge_ps(d, _mm_set1_ps(1f32));
    let x = _mm_select_ps(more_than_one_mask, _mm_div_ps(_mm_set1_ps(1f32), d), d);
    let x2 = _mm_mul_ps(x, x);
    let mut u = _mm_set1_ps(ATAN_POLY_9_F);
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(ATAN_POLY_8_F));
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(ATAN_POLY_7_F));
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(ATAN_POLY_6_F));
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(ATAN_POLY_5_F));
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(ATAN_POLY_4_F));
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(ATAN_POLY_3_F));
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(ATAN_POLY_2_F));
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(ATAN_POLY_1_F));
    u = _mm_mul_ps(u, x);
    u = _mm_select_ps(
        more_than_one_mask,
        _mm_sub_ps(_mm_set1_ps(std::f32::consts::FRAC_PI_2), u),
        u,
    );
    u = _mm_select_ps(negative_mask, _mm_neg_ps(u), u);
    u
}
