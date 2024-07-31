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

use crate::atan::{
    ATAN_POLY_10_D, ATAN_POLY_11_D, ATAN_POLY_12_D, ATAN_POLY_13_D, ATAN_POLY_14_D, ATAN_POLY_15_D,
    ATAN_POLY_16_D, ATAN_POLY_17_D, ATAN_POLY_18_D, ATAN_POLY_19_D, ATAN_POLY_1_D, ATAN_POLY_20_D,
    ATAN_POLY_21_D, ATAN_POLY_2_D, ATAN_POLY_3_D, ATAN_POLY_4_D, ATAN_POLY_5_D, ATAN_POLY_6_D,
    ATAN_POLY_7_D, ATAN_POLY_8_D, ATAN_POLY_9_D,
};
use crate::{_mm256_abs_pd, _mm256_mlaf_pd, _mm256_neg_pd, _mm256_select_pd};

/// Computes Atan function with *ULP 2.0* error
#[inline(always)]
pub unsafe fn _mm256_atan_pd(x: __m256d) -> __m256d {
    let negative_mask = _mm256_cmp_pd::<_CMP_LT_OS>(x, _mm256_setzero_pd());
    let d = _mm256_abs_pd(x);
    let more_than_one_mask = _mm256_cmp_pd::<_CMP_GE_OS>(d, _mm256_set1_pd(1.));
    let x = _mm256_select_pd(more_than_one_mask, _mm256_div_pd(_mm256_set1_pd(1.), d), d);
    let x2 = _mm256_mul_pd(x, x);
    let mut u = _mm256_set1_pd(ATAN_POLY_21_D);
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_20_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_19_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_18_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_17_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_16_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_15_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_14_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_13_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_12_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_11_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_10_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_9_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_8_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_7_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_6_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_5_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_4_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_3_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_2_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(ATAN_POLY_1_D));
    u = _mm256_mul_pd(u, x);
    u = _mm256_select_pd(
        more_than_one_mask,
        _mm256_sub_pd(_mm256_set1_pd(std::f64::consts::FRAC_PI_2), u),
        u,
    );
    u = _mm256_select_pd(negative_mask, _mm256_neg_pd(u), u);
    u
}

#[cfg(test)]
mod tests {
    use crate::avx::general::_mm256_extract_pd;

    use super::*;

    #[test]
    fn test_atand() {
        unsafe {
            let value = _mm256_set1_pd(-2.70752239);
            let comparison = _mm256_atan_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = -1.21699586213983405647952898f64;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm256_set1_pd(2.);
            let comparison = _mm256_atan_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = 1.107148717794090503017065460f64;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm256_set1_pd(-2.);
            let comparison = _mm256_atan_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = -1.107148717794090503017065460f64;
            assert_eq!(flag_1, control);
        }
    }
}
