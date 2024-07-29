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

use crate::{_mm256_cvtepi64_pdx, _mm256_mlaf_pd, _mm256_pow2i_epi64, _mm256_rint_pd, _mm256_select_pd};
use crate::exp::{
    EXP_POLY_10_D, EXP_POLY_1_D, EXP_POLY_2_D, EXP_POLY_3_D, EXP_POLY_4_D, EXP_POLY_5_D,
    EXP_POLY_6_D, EXP_POLY_7_D, EXP_POLY_8_D, EXP_POLY_9_D, L2_L, L2_U, R_LN2,
};

#[inline]
pub unsafe fn _mm256_exp_pd(d: __m256d) -> __m256d {
    let mut r = _mm256_expq_fast_pd(d);
    r = _mm256_select_pd(_mm256_cmp_pd::<_CMP_LT_OS>(d, _mm256_set1_pd(-964f64)), _mm256_setzero_pd(), r);
    r = _mm256_select_pd(
        _mm256_cmp_pd::<_CMP_GT_OS>(d, _mm256_set1_pd(709f64)),
        _mm256_set1_pd(f64::INFINITY),
        r,
    );
    r
}

/// Method that computes exp skipping Inf, Nan checks error bound *ULP 1.0*
#[inline(always)]
pub unsafe fn _mm256_expq_fast_pd(d: __m256d) -> __m256d {
    let q = _mm256_rint_pd(_mm256_mul_pd(d, _mm256_set1_pd(R_LN2)));
    let qf = _mm256_cvtepi64_pdx(q);
    /* exp(x) = 2^i * exp(f); */
    let mut r = _mm256_mlaf_pd(qf, _mm256_set1_pd(-L2_U), d);
    r = _mm256_mlaf_pd(qf, _mm256_set1_pd(-L2_L), r);
    let f = _mm256_mul_pd(r, r);
    let mut u = _mm256_set1_pd(EXP_POLY_10_D);
    u = _mm256_mlaf_pd(u, f, _mm256_set1_pd(EXP_POLY_9_D));
    u = _mm256_mlaf_pd(u, f, _mm256_set1_pd(EXP_POLY_8_D));
    u = _mm256_mlaf_pd(u, f, _mm256_set1_pd(EXP_POLY_7_D));
    u = _mm256_mlaf_pd(u, f, _mm256_set1_pd(EXP_POLY_6_D));
    u = _mm256_mlaf_pd(u, f, _mm256_set1_pd(EXP_POLY_5_D));
    u = _mm256_mlaf_pd(u, f, _mm256_set1_pd(EXP_POLY_4_D));
    u = _mm256_mlaf_pd(u, f, _mm256_set1_pd(EXP_POLY_3_D));
    u = _mm256_mlaf_pd(u, f, _mm256_set1_pd(EXP_POLY_2_D));
    u = _mm256_mlaf_pd(u, f, _mm256_set1_pd(EXP_POLY_1_D));
    let u = _mm256_add_pd(
        _mm256_div_pd(_mm256_mul_pd(r, _mm256_set1_pd(2.)), _mm256_sub_pd(u, r)),
        _mm256_set1_pd(1.),
    );
    let i2 = _mm256_castsi256_pd(_mm256_pow2i_epi64(q));
    let r = _mm256_mul_pd(u, i2);
    r
}

#[cfg(test)]
mod tests {
    use crate::avx::general::_mm256_extract_pd;

    use super::*;

    #[test]
    fn test_expd() {
        unsafe {
            // Test regular
            let value = _mm256_set1_pd(23.);
            let comparison = _mm256_exp_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, 9744803446.248901f64);
        }

        unsafe {
            // Test Infinity
            let value = _mm256_set1_pd(f64::INFINITY);
            let comparison = _mm256_exp_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, f64::INFINITY);
        }

        unsafe {
            // Test Neg Infinity
            let value = _mm256_set1_pd(-15.);
            let comparison = _mm256_exp_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, (-15f64).exp());
        }

        unsafe {
            // Test underflow
            let value = _mm256_set1_pd(-965f64);
            let comparison = _mm256_exp_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1, 0.);
        }

        unsafe {
            // Test overflow
            let value = _mm256_set1_pd(709.5f64);
            let comparison = _mm256_exp_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            assert_eq!(flag_1.is_infinite(), true);
        }
    }
}
