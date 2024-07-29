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

use crate::avx::general::{_mm256_ilogb2k_pd, _mm256_ldexp3k_pd};
use crate::ln::{
    LN_POLY_1_D, LN_POLY_2_D, LN_POLY_3_D, LN_POLY_4_D, LN_POLY_5_D, LN_POLY_6_D, LN_POLY_7_D,
    LN_POLY_8_D,
};
use crate::{
    _mm256_cvtepi64_pdx, _mm256_isinf_pd, _mm256_isnan_pd, _mm256_mlaf_pd, _mm256_neg_epi64,
    _mm256_select_pd,
};

/// Method that computes ln skipping Inf, Nan checks, error bound *ULP 1.5*
#[inline(always)]
pub unsafe fn _mm256_ln_fast_pd(d: __m256d) -> __m256d {
    let n = _mm256_ilogb2k_pd(_mm256_mul_pd(d, _mm256_set1_pd(1. / 0.75)));
    let a = _mm256_ldexp3k_pd(d, _mm256_neg_epi64(n));
    let ones = _mm256_set1_pd(1.);
    let x = _mm256_div_pd(_mm256_sub_pd(a, ones), _mm256_add_pd(a, ones));
    let x2 = _mm256_mul_pd(x, x);
    let mut u = _mm256_set1_pd(LN_POLY_8_D);
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(LN_POLY_7_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(LN_POLY_6_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(LN_POLY_5_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(LN_POLY_4_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(LN_POLY_3_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(LN_POLY_2_D));
    u = _mm256_mlaf_pd(u, x2, _mm256_set1_pd(LN_POLY_1_D));
    let res = _mm256_mlaf_pd(
        _mm256_set1_pd(std::f64::consts::LN_2),
        _mm256_cvtepi64_pdx(n),
        _mm256_mul_pd(x, u),
    );
    res
}

/// Computes natural logarithm for an argument *ULP 1.5*
#[inline]
pub unsafe fn _mm256_ln_pd(d: __m256d) -> __m256d {
    let mut res = _mm256_ln_fast_pd(d);
    // d == 0 || d == Inf -> Inf
    res = _mm256_select_pd(
        _mm256_cmp_pd::<_CMP_EQ_OS>(d, _mm256_setzero_pd()),
        _mm256_set1_pd(f64::NEG_INFINITY),
        res,
    );
    res = _mm256_select_pd(_mm256_isinf_pd(d), _mm256_set1_pd(f64::INFINITY), res);
    // d < 0 || d == Nan -> Nan
    res = _mm256_select_pd(
        _mm256_or_pd(
            _mm256_cmp_pd::<_CMP_LT_OS>(d, _mm256_setzero_pd()),
            _mm256_isnan_pd(d),
        ),
        _mm256_set1_pd(f64::NAN),
        res,
    );
    res
}

#[cfg(test)]
mod tests {
    use crate::avx::general::_mm256_extract_pd;

    use super::*;

    #[test]
    fn test_lnd() {
        unsafe {
            // Test regular
            let value = _mm256_set1_pd(23.);
            let comparison = _mm256_ln_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let diff = 3.1354942159291496908067528318f64
                .to_bits()
                .max(flag_1.to_bits())
                - flag_1
                    .to_bits()
                    .min(3.1354942159291496908067528318f64.to_bits());
            assert!(diff < 5);
        }

        unsafe {
            // Test Infinity
            let value = _mm256_set1_pd(f64::INFINITY);
            let comparison = _mm256_ln_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, f64::INFINITY);
        }

        unsafe {
            // Test Neg Infinity
            let value = _mm256_set1_pd(0.);
            let comparison = _mm256_ln_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, f64::NEG_INFINITY);
        }

        unsafe {
            // Test Infinity, from negatives
            let value = _mm256_set1_pd(-53.);
            let comparison = _mm256_ln_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1.is_nan(), true);
        }
    }
}
