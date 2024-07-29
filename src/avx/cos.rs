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

use crate::sin::{
    PI_A2, PI_B2, SIN_POLY_10_D, SIN_POLY_2_D, SIN_POLY_3_D, SIN_POLY_4_D, SIN_POLY_5_D,
    SIN_POLY_6_D, SIN_POLY_7_D, SIN_POLY_8_D, SIN_POLY_9_D,
};
use crate::{
    _mm256_cvtepi64_pdx, _mm256_mlaf_pd, _mm256_mul_epi64, _mm256_neg_pd, _mm256_rint_pd,
    _mm256_selecti_pd,
};

#[inline]
/// Computes cosine function with error bound *ULP 2.0*
pub unsafe fn _mm256_cos_pd(d: __m256d) -> __m256d {
    let j = _mm256_rint_pd(_mm256_sub_pd(
        _mm256_mul_pd(d, _mm256_set1_pd(std::f64::consts::FRAC_1_PI)),
        _mm256_set1_pd(0.5),
    ));

    let p = _mm256_mul_epi64(j, _mm256_set1_epi64x(2));
    let q = _mm256_add_epi64(_mm256_set1_epi64x(1), p);

    let qf = _mm256_cvtepi64_pdx(q);

    let mut r = _mm256_mlaf_pd(qf, _mm256_set1_pd(-PI_A2 * 0.5), d);
    r = _mm256_mlaf_pd(qf, _mm256_set1_pd(-PI_B2 * 0.5), r);

    let x2 = _mm256_mul_pd(r, r);

    r = _mm256_selecti_pd(
        _mm256_cmpeq_epi64(
            _mm256_and_si256(q, _mm256_set1_epi64x(2)),
            _mm256_setzero_si256(),
        ),
        _mm256_neg_pd(r),
        r,
    );
    let mut res = _mm256_set1_pd(SIN_POLY_10_D);
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(SIN_POLY_9_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(SIN_POLY_8_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(SIN_POLY_7_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(SIN_POLY_6_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(SIN_POLY_5_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(SIN_POLY_4_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(SIN_POLY_3_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(SIN_POLY_2_D));
    res = _mm256_mlaf_pd(res, _mm256_mul_pd(x2, r), r);
    res
}

#[cfg(test)]
mod tests {
    use crate::avx::general::_mm256_extract_pd;

    use super::*;

    #[test]
    fn test_cosd() {
        unsafe {
            let value = _mm256_set1_pd(-2.70752239);
            let comparison = _mm256_cos_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = -0.9072614508830367052983518f64;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm256_set1_pd(2.);
            let comparison = _mm256_cos_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = -0.416146836547142386997568229500f64;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm256_set1_pd(-2.);
            let comparison = _mm256_cos_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = -0.416146836547142386997568229500f64;
            assert_eq!(flag_1, control);
        }
    }
}