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

use crate::lnf::{LN_POLY_1_F, LN_POLY_2_F, LN_POLY_3_F, LN_POLY_4_F, LN_POLY_5_F};
use crate::{
    _mm256_eqzero_ps, _mm256_ilogb2kq_ps, _mm256_isinf_ps, _mm256_isnan_ps, _mm256_ldexp3kq_ps,
    _mm256_ltzero_ps, _mm256_mlaf_ps, _mm256_neg_epi32, _mm256_select_ps,
};

/// Method that computes ln skipping Inf, Nan checks, error bound *ULP 1.5*
#[inline]
pub unsafe fn _mm256_ln_fast_ps(d: __m256) -> __m256 {
    let n = _mm256_ilogb2kq_ps(_mm256_mul_ps(d, _mm256_set1_ps(1f32 / 0.75f32)));
    let a = _mm256_ldexp3kq_ps(d, _mm256_neg_epi32(n));
    let ones = _mm256_set1_ps(1f32);
    let x = _mm256_div_ps(_mm256_sub_ps(a, ones), _mm256_add_ps(a, ones));
    let x2 = _mm256_mul_ps(x, x);
    let mut u = _mm256_set1_ps(LN_POLY_5_F);
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(LN_POLY_4_F));
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(LN_POLY_3_F));
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(LN_POLY_2_F));
    u = _mm256_mlaf_ps(u, x2, _mm256_set1_ps(LN_POLY_1_F));
    let res = _mm256_mlaf_ps(
        _mm256_set1_ps(std::f32::consts::LN_2),
        _mm256_cvtepi32_ps(n),
        _mm256_mul_ps(x, u),
    );
    res
}

/// Computes natural logarithm for an argument *ULP 1.5*
#[inline]
pub unsafe fn _mm256_ln_ps(d: __m256) -> __m256 {
    let mut res = _mm256_ln_fast_ps(d);
    // d == 0 || d == Inf -> Inf
    res = _mm256_select_ps(_mm256_eqzero_ps(d), _mm256_set1_ps(f32::NEG_INFINITY), res);
    res = _mm256_select_ps(_mm256_isinf_ps(d), _mm256_set1_ps(f32::INFINITY), res);
    // d < 0 || d == Nan -> Nan
    res = _mm256_select_ps(
        _mm256_or_ps(_mm256_ltzero_ps(d), _mm256_isnan_ps(d)),
        _mm256_set1_ps(f32::NAN),
        res,
    );
    res
}

#[cfg(test)]
mod tests {
    use crate::_mm256_extract_ps;

    use super::*;

    #[test]
    fn test_ln() {
        unsafe {
            // Test regular
            let value = _mm256_set1_ps(23f32);
            let comparison = _mm256_ln_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, 23f32.ln());
        }

        unsafe {
            // Test Infinity
            let value = _mm256_set1_ps(f32::INFINITY);
            let comparison = _mm256_ln_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, f32::INFINITY);
        }

        unsafe {
            // Test Neg Infinity
            let value = _mm256_set1_ps(0f32);
            let comparison = _mm256_ln_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, f32::NEG_INFINITY);
        }

        unsafe {
            // Test Infinity, from negatives
            let value = _mm256_set1_ps(-53f32);
            let comparison = _mm256_ln_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1.is_nan(), true);
        }
    }
}
