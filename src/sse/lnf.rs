/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::lnf::{LN_POLY_1_F, LN_POLY_2_F, LN_POLY_3_F, LN_POLY_4_F, LN_POLY_5_F};
use crate::{
    _mm_eqzero_ps, _mm_ilogb2kq_ps, _mm_isinf_ps, _mm_isnan_ps, _mm_ldexp3kq_ps, _mm_ltzero_ps,
    _mm_mlaf_ps, _mm_neg_epi32, _mm_select_ps,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Method that computes ln skipping Inf, Nan checks, error bound *ULP 1.5*
#[inline]
pub unsafe fn _mm_ln_fast_ps(d: __m128) -> __m128 {
    let n = _mm_ilogb2kq_ps(_mm_mul_ps(d, _mm_set1_ps(1f32 / 0.75f32)));
    let a = _mm_ldexp3kq_ps(d, _mm_neg_epi32(n));
    let ones = _mm_set1_ps(1f32);
    let x = _mm_div_ps(_mm_sub_ps(a, ones), _mm_add_ps(a, ones));
    let x2 = _mm_mul_ps(x, x);
    let mut u = _mm_set1_ps(LN_POLY_5_F);
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(LN_POLY_4_F));
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(LN_POLY_3_F));
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(LN_POLY_2_F));
    u = _mm_mlaf_ps(u, x2, _mm_set1_ps(LN_POLY_1_F));
    let res = _mm_mlaf_ps(
        _mm_set1_ps(std::f32::consts::LN_2),
        _mm_cvtepi32_ps(n),
        _mm_mul_ps(x, u),
    );
    res
}

/// Computes natural logarithm for an argument *ULP 1.5*
#[inline]
pub unsafe fn _mm_ln_ps(d: __m128) -> __m128 {
    let mut res = _mm_ln_fast_ps(d);
    // d == 0 || d == Inf -> Inf
    res = _mm_select_ps(_mm_eqzero_ps(d), _mm_set1_ps(f32::NEG_INFINITY), res);
    res = _mm_select_ps(_mm_isinf_ps(d), _mm_set1_ps(f32::INFINITY), res);
    // d < 0 || d == Nan -> Nan
    res = _mm_select_ps(
        _mm_or_ps(_mm_ltzero_ps(d), _mm_isnan_ps(d)),
        _mm_set1_ps(f32::NAN),
        res,
    );
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ln() {
        unsafe {
            // Test regular
            let value = _mm_set1_ps(23f32);
            let comparison = _mm_ln_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, 23f32.ln());
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_ps(f32::INFINITY);
            let comparison = _mm_ln_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, f32::INFINITY);
        }

        unsafe {
            // Test Neg Infinity
            let value = _mm_set1_ps(0f32);
            let comparison = _mm_ln_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, f32::NEG_INFINITY);
        }

        unsafe {
            // Test Infinity, from negatives
            let value = _mm_set1_ps(-53f32);
            let comparison = _mm_ln_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1.is_nan(), true);
        }
    }
}
