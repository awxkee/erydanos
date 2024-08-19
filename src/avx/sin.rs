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
    _mm256_cvtepi64_pdx, _mm256_mlaf_pd, _mm256_neg_pd, _mm256_rint_pd, _mm256_selecti_pd,
};

#[inline]
#[target_feature(enable = "avx2")]
/// Computes sine function with *ULP 1.5* on range [-15; 15]
pub unsafe fn _mm256_sin_pd(d: __m256d) -> __m256d {
    let q = _mm256_rint_pd(_mm256_mul_pd(
        d,
        _mm256_set1_pd(std::f64::consts::FRAC_1_PI),
    ));

    let qf = _mm256_cvtepi64_pdx(q);

    let mut r = _mm256_mlaf_pd(qf, _mm256_set1_pd(-PI_A2), d);
    r = _mm256_mlaf_pd(qf, _mm256_set1_pd(-PI_B2), r);

    let x2 = _mm256_mul_pd(r, r);

    r = _mm256_selecti_pd(
        _mm256_cmpeq_epi64(
            _mm256_and_si256(q, _mm256_set1_epi64x(1)),
            _mm256_setzero_si256(),
        ),
        r,
        _mm256_neg_pd(r),
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

    fn bit_diffs(l: f64, o: f64) -> u64 {
        l.to_bits().max(o.to_bits()) - o.to_bits().min(l.to_bits())
    }

    #[test]
    fn test_sind() {
        unsafe {
            let value = _mm256_set1_pd(-2.70752239);
            let comparison = _mm256_sin_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = -0.4205670692548423f64;
            println!("{}", bit_diffs(flag_1, control));
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm256_set1_pd(2.);
            let comparison = _mm256_sin_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = 0.90929742682568169539601f64;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm256_set1_pd(-2.);
            let comparison = _mm256_sin_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = -0.90929742682568169539601f64;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm256_set1_pd(0.);
            let comparison = _mm256_sin_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = 0.;
            assert_eq!(flag_1, control);
        }
    }
}
