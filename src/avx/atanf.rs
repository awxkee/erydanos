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

use crate::atanf::{
    ATAN_POLY_1_F, ATAN_POLY_2_F, ATAN_POLY_3_F, ATAN_POLY_4_F, ATAN_POLY_5_F, ATAN_POLY_6_F,
    ATAN_POLY_7_F, ATAN_POLY_8_F, ATAN_POLY_9_F,
};
use crate::{_mm256_abs_ps, _mm256_ltzero_ps, _mm256_mlaf_ps, _mm256_neg_ps, _mm256_select_ps};

/// Computes Atan function with *ULP 1.0* error
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_atan_ps(x: __m256) -> __m256 {
    let negative_mask = _mm256_ltzero_ps(x);
    let d = _mm256_abs_ps(x);
    let more_than_one_mask = _mm256_cmp_ps::<_CMP_GE_OS>(d, _mm256_set1_ps(1f32));
    let x = _mm256_select_ps(
        more_than_one_mask,
        _mm256_div_ps(_mm256_set1_ps(1f32), d),
        d,
    );
    let x2 = _mm256_mul_ps(x, x);
    let mut u = _mm256_set1_ps(ATAN_POLY_9_F);
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(ATAN_POLY_8_F));
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(ATAN_POLY_7_F));
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(ATAN_POLY_6_F));
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(ATAN_POLY_5_F));
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(ATAN_POLY_4_F));
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(ATAN_POLY_3_F));
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(ATAN_POLY_2_F));
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(ATAN_POLY_1_F));
    u = _mm256_mul_ps(u, x);
    u = _mm256_select_ps(
        more_than_one_mask,
        _mm256_sub_ps(_mm256_set1_ps(std::f32::consts::FRAC_PI_2), u),
        u,
    );
    u = _mm256_select_ps(negative_mask, _mm256_neg_ps(u), u);
    u
}

#[cfg(test)]
mod tests {
    use crate::_mm256_extract_ps;

    use super::*;

    #[test]
    fn test_atanf() {
        unsafe {
            let value = _mm256_set1_ps(-2.70752239);
            let comparison = _mm256_atan_ps(value);
            let flag_1 = _mm256_extract_ps::<1>(comparison);
            let control = -1.216996f32;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm256_set1_ps(2f32);
            let comparison = _mm256_atan_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            let control = 1.107148717794090503017065460f32;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm256_set1_ps(-2f32);
            let comparison = _mm256_atan_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            let control = -1.107148717794090503017065460f32;
            assert_eq!(flag_1, control);
        }
    }
}
