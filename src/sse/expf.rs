/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::expf::{
    EXP_POLY_1_S, EXP_POLY_2_S, EXP_POLY_3_S, EXP_POLY_4_S, EXP_POLY_5_S, L2L_F, L2U_F,
};
use crate::sse::generalf::_mm_rint_ps;
use crate::{_mm_mlaf_ps, _mm_pow2if_epi32, _mm_select_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes exp for an argument *ULP 1.0*
#[inline]
pub unsafe fn _mm_exp_ps(d: __m128) -> __m128 {
    let mut r = _mm_exp_fast_ps(d);
    r = _mm_select_ps(_mm_cmplt_ps(d, _mm_set1_ps(-87f32)), _mm_setzero_ps(), r);
    r = _mm_select_ps(
        _mm_cmpgt_ps(d, _mm_set1_ps(88f32)),
        _mm_set1_ps(f32::INFINITY),
        r,
    );
    r
}

/// Method that computes exp skipping Inf, Nan checks error bound *ULP 1.0*
#[inline]
pub unsafe fn _mm_exp_fast_ps(d: __m128) -> __m128 {
    let q = _mm_rint_ps(_mm_mul_ps(d, _mm_set1_ps(std::f32::consts::LOG2_E)));
    let qf = _mm_cvtepi32_ps(q);
    /* exp(x) = 2^i * exp(f); */
    let mut r = _mm_mlaf_ps(qf, _mm_set1_ps(-L2U_F), d);
    r = _mm_mlaf_ps(qf, _mm_set1_ps(-L2L_F), r);
    let f = _mm_mul_ps(r, r);
    let mut u = _mm_set1_ps(EXP_POLY_5_S);
    u = _mm_mlaf_ps(u, f, _mm_set1_ps(EXP_POLY_4_S));
    u = _mm_mlaf_ps(u, f, _mm_set1_ps(EXP_POLY_3_S));
    u = _mm_mlaf_ps(u, f, _mm_set1_ps(EXP_POLY_2_S));
    u = _mm_mlaf_ps(u, f, _mm_set1_ps(EXP_POLY_1_S));
    let u = _mm_add_ps(
        _mm_div_ps(_mm_mul_ps(r, _mm_set1_ps(2f32)), _mm_sub_ps(u, r)),
        _mm_set1_ps(1f32),
    );
    let i2 = _mm_castsi128_ps(_mm_pow2if_epi32(q));
    let r = _mm_mul_ps(u, i2);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expf() {
        unsafe {
            // Test regular
            let value = _mm_set1_ps(23f32);
            let comparison = _mm_exp_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, 23f32.exp());
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_ps(f32::INFINITY);
            let comparison = _mm_exp_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, f32::INFINITY);
        }

        unsafe {
            // Test Neg Infinity
            let value = _mm_set1_ps(-15f32);
            let comparison = _mm_exp_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, (-15f32).exp());
        }

        unsafe {
            // Test underflow
            let value = _mm_set1_ps(-89f32);
            let comparison = _mm_exp_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1, 0f32);
        }

        unsafe {
            // Test overflow
            let value = _mm_set1_ps(89f32);
            let comparison = _mm_exp_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag_1.is_infinite(), true);
        }
    }
}
