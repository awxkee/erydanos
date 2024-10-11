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
use crate::{_mm_abs_pd, _mm_copysign_pd, _mm_mlaf_pd, _mm_select_pd};

/// Computes arcsin, error bound *ULP 2.0*
#[inline]
pub unsafe fn _mm_asin_pd(d: __m128d) -> __m128d {
    let ones = _mm_set1_pd(1.);
    let ca = _mm_abs_pd(d);
    let nan_mask = _mm_cmpgt_pd(ca, ones);
    // for more 0.5
    let reverse_05_mask = _mm_cmpge_pd(ca, _mm_set1_pd(0.5f64));
    let reversed = _mm_sqrt_pd(_mm_div_pd(_mm_sub_pd(ones, ca), _mm_set1_pd(2.)));
    let x = _mm_select_pd(reverse_05_mask, reversed, ca);
    let zeros_is_zeros = _mm_cmpeq_pd(d, _mm_setzero_pd());
    let mut u = _mm_set1_pd(ASIN_POLY_19_D);
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_18_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_17_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_16_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_15_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_14_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_13_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_12_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_11_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_10_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_9_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_8_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_7_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_6_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_5_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_4_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_3_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_2_D));
    u = _mm_mlaf_pd(u, x, _mm_set1_pd(ASIN_POLY_1_D));
    u = _mm_mul_pd(u, x);
    let j = u;
    let reconstruct_reversed = _mm_select_pd(
        _mm_set1_pd(-2f64),
        j,
        _mm_set1_pd(std::f64::consts::FRAC_PI_2),
    );
    let mut ret = _mm_select_pd(reverse_05_mask, reconstruct_reversed, j);
    ret = _mm_select_pd(nan_mask, _mm_set1_pd(f64::NAN), ret);
    ret = _mm_select_pd(zeros_is_zeros, _mm_set1_pd(0.), ret);
    _mm_copysign_pd(ret, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_mm_extract_pd;

    #[test]
    fn test_asind() {
        unsafe {
            let value = _mm_set1_pd(0.3);
            let comparison = _mm_asin_pd(value);
            let flag_1 = _mm_extract_pd::<1>(comparison);
            let control = 0.30469265401539747f64;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm_set1_pd(-0.3);
            let comparison = _mm_asin_pd(value);
            let flag_1 = _mm_extract_pd::<1>(comparison);
            let control = -0.30469265401539747f64;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm_set1_pd(-2f64);
            let comparison = _mm_asin_pd(value);
            let flag_1 = _mm_extract_pd::<1>(comparison);
            assert!(flag_1.is_nan());
        }
    }
}
