/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::cosf::{PI_A_F, PI_B_F, PI_C_F, PI_D_F};
use crate::sinf::{SIN_POLY_1_S, SIN_POLY_2_S, SIN_POLY_3_S, SIN_POLY_4_S, SIN_POLY_5_S};
use crate::{_mm_mlaf_ps, _mm_neg_ps, _mm_rint_ps, _mm_selecti_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes sine function with error bound *ULP 1.5*
#[inline]
pub unsafe fn _mm_sin_ps(d: __m128) -> __m128 {
    let q = _mm_rint_ps(_mm_mul_ps(d, _mm_set1_ps(std::f32::consts::FRAC_1_PI)));
    let qf = _mm_cvtepi32_ps(q);

    let mut r = _mm_mlaf_ps(qf, _mm_set1_ps(-PI_A_F), d);
    r = _mm_mlaf_ps(qf, _mm_set1_ps(-PI_B_F), r);
    r = _mm_mlaf_ps(qf, _mm_set1_ps(-PI_C_F), r);
    r = _mm_mlaf_ps(qf, _mm_set1_ps(-PI_D_F), r);

    let x2 = _mm_mul_ps(r, r);

    r = _mm_selecti_ps(
        _mm_cmpeq_epi32(_mm_and_si128(q, _mm_set1_epi32(1)), _mm_setzero_si128()),
        r,
        _mm_neg_ps(r),
    );
    let mut res = _mm_set1_ps(SIN_POLY_5_S);
    res = _mm_mlaf_ps(res, x2, _mm_set1_ps(SIN_POLY_4_S));
    res = _mm_mlaf_ps(res, x2, _mm_set1_ps(SIN_POLY_3_S));
    res = _mm_mlaf_ps(res, x2, _mm_set1_ps(SIN_POLY_2_S));
    res = _mm_mlaf_ps(res, x2, _mm_set1_ps(SIN_POLY_1_S));
    res = _mm_mlaf_ps(res, _mm_mul_ps(x2, r), r);
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinf() {
        unsafe {
            let value = _mm_set1_ps(2f32);
            let comparison = _mm_sin_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            let control = 0.9092974268256f32;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm_set1_ps(-2f32);
            let comparison = _mm_sin_ps(value);
            let flag_1 = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            let control = -0.9092974268256f32;
            assert_eq!(flag_1, control);
        }
    }
}
