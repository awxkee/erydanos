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

use crate::asinf::{
    ASIN_POLY_2_F, ASIN_POLY_3_F, ASIN_POLY_4_F, ASIN_POLY_5_F, ASIN_POLY_6_F, ASIN_POLY_7_F,
    ASIN_POLY_8_F, ASIN_POLY_9_F,
};
use crate::avx::generalf::_mm256_copysign_ps;
use crate::{_mm256_abs_ps, _mm256_mlaf_ps, _mm256_select_ps};

/// Computes arcsin, error bound *ULP 2.0*
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_asin_ps(d: __m256) -> __m256 {
    let ones = _mm256_set1_ps(1f32);
    let ca = _mm256_abs_ps(d);
    let nan_mask = _mm256_cmp_ps::<_CMP_GT_OS>(ca, ones);
    // for more 0.5
    let reverse_05_mask = _mm256_cmp_ps::<_CMP_GE_OS>(ca, _mm256_set1_ps(0.5f32));
    let reversed = _mm256_sqrt_ps(_mm256_div_ps(_mm256_sub_ps(ones, ca), _mm256_set1_ps(2f32)));
    let x = _mm256_select_ps(reverse_05_mask, reversed, ca);
    let zeros_is_zeros = _mm256_cmp_ps::<_CMP_EQ_OS>(d, _mm256_setzero_ps());
    let mut u = _mm256_castsi256_ps(_mm256_set1_epi32(ASIN_POLY_9_F as i32));
    u = _mm256_mlaf_ps(
        u,
        x,
        _mm256_castsi256_ps(_mm256_set1_epi32(ASIN_POLY_8_F as i32)),
    );
    u = _mm256_mlaf_ps(
        u,
        x,
        _mm256_castsi256_ps(_mm256_set1_epi32(ASIN_POLY_7_F as i32)),
    );
    u = _mm256_mlaf_ps(
        u,
        x,
        _mm256_castsi256_ps(_mm256_set1_epi32(ASIN_POLY_6_F as i32)),
    );
    u = _mm256_mlaf_ps(
        u,
        x,
        _mm256_castsi256_ps(_mm256_set1_epi32(ASIN_POLY_5_F as i32)),
    );
    u = _mm256_mlaf_ps(
        u,
        x,
        _mm256_castsi256_ps(_mm256_set1_epi32(ASIN_POLY_4_F as i32)),
    );
    u = _mm256_mlaf_ps(
        u,
        x,
        _mm256_castsi256_ps(_mm256_set1_epi32(ASIN_POLY_3_F as i32)),
    );
    u = _mm256_mlaf_ps(
        u,
        x,
        _mm256_castsi256_ps(_mm256_set1_epi32(ASIN_POLY_2_F as i32)),
    );
    u = _mm256_mul_ps(u, x);
    let j = u;
    let reconstruct_reversed = _mm256_select_ps(
        _mm256_set1_ps(-2f32),
        j,
        _mm256_set1_ps(std::f32::consts::FRAC_PI_2),
    );
    let mut ret = _mm256_select_ps(reverse_05_mask, reconstruct_reversed, j);
    ret = _mm256_select_ps(nan_mask, _mm256_set1_ps(f32::NAN), ret);
    ret = _mm256_select_ps(zeros_is_zeros, _mm256_set1_ps(0f32), ret);
    _mm256_copysign_ps(ret, d)
}

#[cfg(test)]
mod tests {
    use crate::_mm256_extract_ps;

    use super::*;

    #[test]
    fn test_asinf() {
        unsafe {
            let value = _mm256_set1_ps(0.3);
            let comparison = _mm256_asin_ps(value);
            let flag_1 = _mm256_extract_ps::<1>(comparison);
            let control = 0.304692654015397507972f32;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm256_set1_ps(-0.3);
            let comparison = _mm256_asin_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            let control = -0.304692654015397507972f32;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm256_set1_ps(-2f32);
            let comparison = _mm256_asin_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert!(flag_1.is_nan());
        }
    }
}
