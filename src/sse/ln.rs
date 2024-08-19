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

use crate::ln::{
    LN_POLY_1_D, LN_POLY_2_D, LN_POLY_3_D, LN_POLY_4_D, LN_POLY_5_D, LN_POLY_6_D, LN_POLY_7_D,
    LN_POLY_8_D,
};
use crate::sse::general::{_mm_ilogb2k_pd, _mm_ldexp3k_pd, _mm_ltzero_pd};
use crate::{
    _mm_cvtepi64_pd, _mm_eqzero_pd, _mm_isinf_pd, _mm_isnan_pd, _mm_mlaf_pd, _mm_neg_epi64,
    _mm_select_pd,
};

/// Method that computes ln skipping Inf, Nan checks, error bound *ULP 1.5*
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_ln_fast_pd(d: __m128d) -> __m128d {
    let n = _mm_ilogb2k_pd(_mm_mul_pd(d, _mm_set1_pd(1. / 0.75)));
    let a = _mm_ldexp3k_pd(d, _mm_neg_epi64(n));
    let ones = _mm_set1_pd(1.);
    let x = _mm_div_pd(_mm_sub_pd(a, ones), _mm_add_pd(a, ones));
    let x2 = _mm_mul_pd(x, x);
    let mut u = _mm_set1_pd(LN_POLY_8_D);
    u = _mm_mlaf_pd(u, x2, _mm_set1_pd(LN_POLY_7_D));
    u = _mm_mlaf_pd(u, x2, _mm_set1_pd(LN_POLY_6_D));
    u = _mm_mlaf_pd(u, x2, _mm_set1_pd(LN_POLY_5_D));
    u = _mm_mlaf_pd(u, x2, _mm_set1_pd(LN_POLY_4_D));
    u = _mm_mlaf_pd(u, x2, _mm_set1_pd(LN_POLY_3_D));
    u = _mm_mlaf_pd(u, x2, _mm_set1_pd(LN_POLY_2_D));
    u = _mm_mlaf_pd(u, x2, _mm_set1_pd(LN_POLY_1_D));
    let res = _mm_mlaf_pd(
        _mm_set1_pd(std::f64::consts::LN_2),
        _mm_cvtepi64_pd(n),
        _mm_mul_pd(x, u),
    );
    res
}

/// Computes natural logarithm for an argument *ULP 1.5*
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_ln_pd(d: __m128d) -> __m128d {
    let mut res = _mm_ln_fast_pd(d);
    // d == 0 || d == Inf -> Inf
    res = _mm_select_pd(_mm_eqzero_pd(d), _mm_set1_pd(f64::NEG_INFINITY), res);
    res = _mm_select_pd(_mm_isinf_pd(d), _mm_set1_pd(f64::INFINITY), res);
    // d < 0 || d == Nan -> Nan
    res = _mm_select_pd(
        _mm_or_pd(_mm_ltzero_pd(d), _mm_isnan_pd(d)),
        _mm_set1_pd(f64::NAN),
        res,
    );
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_mm_extract_pd;

    #[test]
    fn test_lnd() {
        unsafe {
            // Test regular
            let value = _mm_set1_pd(23.);
            let comparison = _mm_ln_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
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
            let value = _mm_set1_pd(f64::INFINITY);
            let comparison = _mm_ln_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, f64::INFINITY);
        }

        unsafe {
            // Test Neg Infinity
            let value = _mm_set1_pd(0.);
            let comparison = _mm_ln_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, f64::NEG_INFINITY);
        }

        unsafe {
            // Test Infinity, from negatives
            let value = _mm_set1_pd(-53.);
            let comparison = _mm_ln_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1.is_nan(), true);
        }
    }
}
