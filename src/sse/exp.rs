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

use crate::exp::{
    EXP_POLY_10_D, EXP_POLY_1_D, EXP_POLY_2_D, EXP_POLY_3_D, EXP_POLY_4_D, EXP_POLY_5_D,
    EXP_POLY_6_D, EXP_POLY_7_D, EXP_POLY_8_D, EXP_POLY_9_D, L2_L, L2_U, R_LN2,
};
use crate::sse::general::{_mm_pow2i_epi64, _mm_rint_pd};
use crate::{_mm_cvtepi64_pd, _mm_mlaf_pd, _mm_select_pd};

/// Computes exp for an argument *ULP 2.0*
#[inline]
pub unsafe fn _mm_exp_pd(d: __m128d) -> __m128d {
    let mut r = _mm_exp_fast_pd(d);
    r = _mm_select_pd(_mm_cmplt_pd(d, _mm_set1_pd(-964f64)), _mm_setzero_pd(), r);
    r = _mm_select_pd(
        _mm_cmpgt_pd(d, _mm_set1_pd(709f64)),
        _mm_set1_pd(f64::INFINITY),
        r,
    );
    r
}

/// Method that computes exp skipping Inf, Nan checks error bound *ULP 2.0*
#[inline]
pub unsafe fn _mm_exp_fast_pd(d: __m128d) -> __m128d {
    let q = _mm_rint_pd(_mm_mul_pd(d, _mm_set1_pd(R_LN2)));
    let qf = _mm_cvtepi64_pd(q);
    /* exp(x) = 2^i * exp(f); */
    let mut r = _mm_mlaf_pd(qf, _mm_set1_pd(-L2_U), d);
    r = _mm_mlaf_pd(qf, _mm_set1_pd(-L2_L), r);
    let f = _mm_mul_pd(r, r);
    let mut u = _mm_set1_pd(EXP_POLY_10_D);
    u = _mm_mlaf_pd(u, f, _mm_set1_pd(EXP_POLY_9_D));
    u = _mm_mlaf_pd(u, f, _mm_set1_pd(EXP_POLY_8_D));
    u = _mm_mlaf_pd(u, f, _mm_set1_pd(EXP_POLY_7_D));
    u = _mm_mlaf_pd(u, f, _mm_set1_pd(EXP_POLY_6_D));
    u = _mm_mlaf_pd(u, f, _mm_set1_pd(EXP_POLY_5_D));
    u = _mm_mlaf_pd(u, f, _mm_set1_pd(EXP_POLY_4_D));
    u = _mm_mlaf_pd(u, f, _mm_set1_pd(EXP_POLY_3_D));
    u = _mm_mlaf_pd(u, f, _mm_set1_pd(EXP_POLY_2_D));
    u = _mm_mlaf_pd(u, f, _mm_set1_pd(EXP_POLY_1_D));
    let u = _mm_add_pd(
        _mm_div_pd(_mm_mul_pd(r, _mm_set1_pd(2.)), _mm_sub_pd(u, r)),
        _mm_set1_pd(1.),
    );
    let i2 = _mm_castsi128_pd(_mm_pow2i_epi64(q));
    let r = _mm_mul_pd(u, i2);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_mm_extract_pd;

    #[test]
    fn test_expd() {
        unsafe {
            // Test regular
            let value = _mm_set1_pd(23.);
            let comparison = _mm_exp_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, 9744803446.248901f64);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_pd(f64::INFINITY);
            let comparison = _mm_exp_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, f64::INFINITY);
        }

        unsafe {
            // Test Neg Infinity
            let value = _mm_set1_pd(-15.);
            let comparison = _mm_exp_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, (-15f64).exp());
        }

        unsafe {
            // Test underflow
            let value = _mm_set1_pd(-965f64);
            let comparison = _mm_exp_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1, 0.);
        }

        unsafe {
            // Test overflow
            let value = _mm_set1_pd(709.5f64);
            let comparison = _mm_exp_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            assert_eq!(flag_1.is_infinite(), true);
        }
    }
}
