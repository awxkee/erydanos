/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::asinf::{
    ASIN_POLY_2_F, ASIN_POLY_3_F, ASIN_POLY_4_F, ASIN_POLY_5_F, ASIN_POLY_6_F, ASIN_POLY_7_F,
    ASIN_POLY_8_F, ASIN_POLY_9_F,
};
use crate::{_mm_abs_ps, _mm_copysign_ps, _mm_mlaf_ps, _mm_select_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes arcsin, error bound *ULP 2.0*
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_asin_ps(d: __m128) -> __m128 {
    let ones = _mm_set1_ps(1f32);
    let ca = _mm_abs_ps(d);
    let nan_mask = _mm_cmpgt_ps(ca, ones);
    // for more 0.5
    let reverse_05_mask = _mm_cmpge_ps(ca, _mm_set1_ps(0.5f32));
    let reversed = _mm_sqrt_ps(_mm_div_ps(_mm_sub_ps(ones, ca), _mm_set1_ps(2f32)));
    let x = _mm_select_ps(reverse_05_mask, reversed, ca);
    let zeros_is_zeros = _mm_cmpeq_ps(d, _mm_setzero_ps());
    let mut u = _mm_castsi128_ps(_mm_set1_epi32(ASIN_POLY_9_F as i32));
    u = _mm_mlaf_ps(u, x, _mm_castsi128_ps(_mm_set1_epi32(ASIN_POLY_8_F as i32)));
    u = _mm_mlaf_ps(u, x, _mm_castsi128_ps(_mm_set1_epi32(ASIN_POLY_7_F as i32)));
    u = _mm_mlaf_ps(u, x, _mm_castsi128_ps(_mm_set1_epi32(ASIN_POLY_6_F as i32)));
    u = _mm_mlaf_ps(u, x, _mm_castsi128_ps(_mm_set1_epi32(ASIN_POLY_5_F as i32)));
    u = _mm_mlaf_ps(u, x, _mm_castsi128_ps(_mm_set1_epi32(ASIN_POLY_4_F as i32)));
    u = _mm_mlaf_ps(u, x, _mm_castsi128_ps(_mm_set1_epi32(ASIN_POLY_3_F as i32)));
    u = _mm_mlaf_ps(u, x, _mm_castsi128_ps(_mm_set1_epi32(ASIN_POLY_2_F as i32)));
    u = _mm_mul_ps(u, x);
    let j = u;
    let reconstruct_reversed = _mm_select_ps(
        _mm_set1_ps(-2f32),
        j,
        _mm_set1_ps(std::f32::consts::FRAC_PI_2),
    );
    let mut ret = _mm_select_ps(reverse_05_mask, reconstruct_reversed, j);
    ret = _mm_select_ps(nan_mask, _mm_set1_ps(f32::NAN), ret);
    ret = _mm_select_ps(zeros_is_zeros, _mm_set1_ps(0f32), ret);
    _mm_copysign_ps(ret, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asinf() {
        unsafe {
            let value = _mm_set1_ps(0.3);
            let comparison = _mm_asin_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<1>(comparison) as u32);
            let control = 0.304692654015397507972f32;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm_set1_ps(-0.3);
            let comparison = _mm_asin_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            let control = -0.304692654015397507972f32;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm_set1_ps(-2f32);
            let comparison = _mm_asin_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert!(flag_1.is_nan());
        }
    }
}
