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

use crate::asin::{
    ASIN_POLY_10_D, ASIN_POLY_11_D, ASIN_POLY_12_D, ASIN_POLY_13_D, ASIN_POLY_14_D, ASIN_POLY_15_D,
    ASIN_POLY_16_D, ASIN_POLY_17_D, ASIN_POLY_18_D, ASIN_POLY_19_D, ASIN_POLY_1_D, ASIN_POLY_2_D,
    ASIN_POLY_3_D, ASIN_POLY_4_D, ASIN_POLY_5_D, ASIN_POLY_6_D, ASIN_POLY_7_D, ASIN_POLY_8_D,
    ASIN_POLY_9_D,
};
use crate::{_mm256_abs_pd, _mm256_copysign_pd, _mm256_mlaf_pd, _mm256_select_pd};

/// Computes arcsin, error bound *ULP 2.0*
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_asin_pd(d: __m256d) -> __m256d {
    let ones = _mm256_set1_pd(1.);
    let ca = _mm256_abs_pd(d);
    let nan_mask = _mm256_cmp_pd::<_CMP_GT_OS>(ca, ones);
    // for more 0.5
    let reverse_05_mask = _mm256_cmp_pd::<_CMP_GE_OS>(ca, _mm256_set1_pd(0.5f64));
    let reversed = _mm256_sqrt_pd(_mm256_div_pd(_mm256_sub_pd(ones, ca), _mm256_set1_pd(2.)));
    let x = _mm256_select_pd(reverse_05_mask, reversed, ca);
    let zeros_is_zeros = _mm256_cmp_pd::<_CMP_EQ_OS>(d, _mm256_setzero_pd());
    let mut u = _mm256_set1_pd(ASIN_POLY_19_D);
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_18_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_17_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_16_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_15_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_14_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_13_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_12_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_11_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_10_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_9_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_8_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_7_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_6_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_5_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_4_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_3_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_2_D));
    u = _mm256_mlaf_pd(u, x, _mm256_set1_pd(ASIN_POLY_1_D));
    u = _mm256_mul_pd(u, x);
    let j = u;
    let reconstruct_reversed = _mm256_select_pd(
        _mm256_set1_pd(-2f64),
        j,
        _mm256_set1_pd(std::f64::consts::FRAC_PI_2),
    );
    let mut ret = _mm256_select_pd(reverse_05_mask, reconstruct_reversed, j);
    ret = _mm256_select_pd(nan_mask, _mm256_set1_pd(f64::NAN), ret);
    ret = _mm256_select_pd(zeros_is_zeros, _mm256_set1_pd(0.), ret);
    _mm256_copysign_pd(ret, d)
}

#[cfg(test)]
mod tests {
    use crate::_mm256_extract_pd;

    use super::*;

    #[test]
    fn test_asind() {
        unsafe {
            let value = _mm256_set1_pd(0.3);
            let comparison = _mm256_asin_pd(value);
            let flag_1 = _mm256_extract_pd::<1>(comparison);
            let control = 0.30469265401539747f64;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm256_set1_pd(-0.3);
            let comparison = _mm256_asin_pd(value);
            let flag_1 = _mm256_extract_pd::<1>(comparison);
            let control = -0.30469265401539747f64;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm256_set1_pd(-2f64);
            let comparison = _mm256_asin_pd(value);
            let flag_1 = _mm256_extract_pd::<1>(comparison);
            assert!(flag_1.is_nan());
        }
    }
}
