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

use crate::cosf::{PI_A_F, PI_B_F, PI_C_F, PI_D_F};
use crate::sinf::{SIN_POLY_1_S, SIN_POLY_2_S, SIN_POLY_3_S, SIN_POLY_4_S, SIN_POLY_5_S};
use crate::{_mm256_mlaf_ps, _mm256_neg_ps, _mm256_rint_ps, _mm256_selecti_ps};

#[inline]
/// Computes cosine function with error bound *ULP 1.5*
pub unsafe fn _mm256_cos_ps(d: __m256) -> __m256 {
    let q = _mm256_add_epi32(
        _mm256_set1_epi32(1),
        _mm256_mullo_epi32(
            _mm256_rint_ps(_mm256_sub_ps(
                _mm256_mul_ps(d, _mm256_set1_ps(std::f32::consts::FRAC_1_PI)),
                _mm256_set1_ps(0.5f32),
            )),
            _mm256_set1_epi32(2),
        ),
    );
    let qf = _mm256_cvtepi32_ps(q);

    let mut r = _mm256_mlaf_ps(qf, _mm256_set1_ps(-PI_A_F * 0.5), d);
    r = _mm256_mlaf_ps(qf, _mm256_set1_ps(-PI_B_F * 0.5), r);
    r = _mm256_mlaf_ps(qf, _mm256_set1_ps(-PI_C_F * 0.5), r);
    r = _mm256_mlaf_ps(qf, _mm256_set1_ps(-PI_D_F * 0.5), r);

    let x2 = _mm256_mul_ps(r, r);

    r = _mm256_selecti_ps(
        _mm256_cmpeq_epi32(
            _mm256_and_si256(q, _mm256_set1_epi32(2)),
            _mm256_setzero_si256(),
        ),
        _mm256_neg_ps(r),
        r,
    );
    let mut res = _mm256_set1_ps(SIN_POLY_5_S);
    res = _mm256_mlaf_ps(res, x2, _mm256_set1_ps(SIN_POLY_4_S));
    res = _mm256_mlaf_ps(res, x2, _mm256_set1_ps(SIN_POLY_3_S));
    res = _mm256_mlaf_ps(res, x2, _mm256_set1_ps(SIN_POLY_2_S));
    res = _mm256_mlaf_ps(res, x2, _mm256_set1_ps(SIN_POLY_1_S));
    res = _mm256_mlaf_ps(res, _mm256_mul_ps(x2, r), r);
    res
}

#[cfg(test)]
mod tests {
    use crate::_mm256_extract_ps;

    use super::*;

    #[test]
    fn test_cosf() {
        unsafe {
            let value = _mm256_set1_ps(-2.70752239);
            let comparison = _mm256_cos_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            let control = -0.907261451f32;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm256_set1_ps(2f32);
            let comparison = _mm256_cos_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            let control = -0.4161468;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm256_set1_ps(-2f32);
            let comparison = _mm256_cos_ps(value);
            let flag_1 = _mm256_extract_ps::<0>(comparison);
            let control = -0.4161468;
            assert_eq!(flag_1, control);
        }
    }
}
