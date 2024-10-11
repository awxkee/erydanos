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

use crate::expf::{
    EXP_POLY_1_S, EXP_POLY_2_S, EXP_POLY_3_S, EXP_POLY_4_S, EXP_POLY_5_S, L2L_F, L2U_F,
};
use crate::{_mm256_mlaf_ps, _mm256_pow2if_epi32, _mm256_rint_ps, _mm256_select_ps};

/// Computes exp for an argument *ULP 1.0*
#[inline]
pub unsafe fn _mm256_exp_ps(d: __m256) -> __m256 {
    let mut r = _mm256_expq_fast_ps(d);
    r = _mm256_select_ps(
        _mm256_cmp_ps::<_CMP_LT_OS>(d, _mm256_set1_ps(-87f32)),
        _mm256_setzero_ps(),
        r,
    );
    r = _mm256_select_ps(
        _mm256_cmp_ps::<_CMP_GT_OS>(d, _mm256_set1_ps(88f32)),
        _mm256_set1_ps(f32::INFINITY),
        r,
    );
    r
}

/// Method that computes exp skipping Inf, Nan checks error bound *ULP 1.0*
#[inline]
pub unsafe fn _mm256_expq_fast_ps(d: __m256) -> __m256 {
    let q = _mm256_rint_ps(_mm256_mul_ps(d, _mm256_set1_ps(std::f32::consts::LOG2_E)));
    let qf = _mm256_cvtepi32_ps(q);
    /* exp(x) = 2^i * exp(f); */
    let mut r = _mm256_mlaf_ps(qf, _mm256_set1_ps(-L2U_F), d);
    r = _mm256_mlaf_ps(qf, _mm256_set1_ps(-L2L_F), r);
    let f = _mm256_mul_ps(r, r);
    let mut u = _mm256_set1_ps(EXP_POLY_5_S);
    u = _mm256_mlaf_ps(u, f, _mm256_set1_ps(EXP_POLY_4_S));
    u = _mm256_mlaf_ps(u, f, _mm256_set1_ps(EXP_POLY_3_S));
    u = _mm256_mlaf_ps(u, f, _mm256_set1_ps(EXP_POLY_2_S));
    u = _mm256_mlaf_ps(u, f, _mm256_set1_ps(EXP_POLY_1_S));
    let u = _mm256_add_ps(
        _mm256_div_ps(_mm256_mul_ps(r, _mm256_set1_ps(2f32)), _mm256_sub_ps(u, r)),
        _mm256_set1_ps(1f32),
    );
    let i2 = _mm256_castsi256_ps(_mm256_pow2if_epi32(q));
    let r = _mm256_mul_ps(u, i2);
    r
}

#[cfg(test)]
mod tests {
    use crate::_mm256_extract_ps;

    use super::*;

    #[test]
    fn test_expf() {
        unsafe {
            // Test regular
            let value = _mm256_set1_ps(23f32);
            let comparison = _mm256_exp_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, 23f32.exp());
        }

        unsafe {
            // Test Infinity
            let value = _mm256_set1_ps(f32::INFINITY);
            let comparison = _mm256_exp_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, f32::INFINITY);
        }

        unsafe {
            // Test Neg Infinity
            let value = _mm256_set1_ps(-15f32);
            let comparison = _mm256_exp_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, (-15f32).exp());
        }

        unsafe {
            // Test underflow
            let value = _mm256_set1_ps(-89f32);
            let comparison = _mm256_exp_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1, 0f32);
        }

        unsafe {
            // Test overflow
            let value = _mm256_set1_ps(89f32);
            let comparison = _mm256_exp_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            assert_eq!(flag_1.is_infinite(), true);
        }
    }
}
