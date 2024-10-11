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

use crate::sin::{PI_A2, PI_B2};
use crate::tan::{
    TAN_POLY_1_D, TAN_POLY_2_D, TAN_POLY_3_D, TAN_POLY_4_D, TAN_POLY_5_D, TAN_POLY_6_D,
    TAN_POLY_7_D, TAN_POLY_8_D, TAN_POLY_9_D,
};
use crate::{
    _mm256_cvtepi64_pdx, _mm256_mlaf_pd, _mm256_neg_pd, _mm256_rint_pd, _mm256_selecti_pd,
};

#[inline]
/// Computes tan function with error bound *ULP 1.5*
pub unsafe fn _mm256_tan_pd(d: __m256d) -> __m256d {
    let q = _mm256_rint_pd(_mm256_mul_pd(
        d,
        _mm256_set1_pd(std::f64::consts::FRAC_2_PI),
    ));
    let qf = _mm256_cvtepi64_pdx(q);

    let mut r = _mm256_mlaf_pd(qf, _mm256_set1_pd(-PI_A2 * 0.5), d);
    r = _mm256_mlaf_pd(qf, _mm256_set1_pd(-PI_B2 * 0.5), r);

    let is_even = _mm256_cmpeq_epi64(
        _mm256_and_si256(q, _mm256_set1_epi64x(1)),
        _mm256_setzero_si256(),
    );

    r = _mm256_selecti_pd(is_even, r, _mm256_neg_pd(r));

    r = _mm256_mul_pd(r, _mm256_set1_pd(0.5));

    let x2 = _mm256_mul_pd(r, r);

    let mut res = _mm256_set1_pd(TAN_POLY_9_D);
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(TAN_POLY_8_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(TAN_POLY_7_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(TAN_POLY_6_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(TAN_POLY_5_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(TAN_POLY_4_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(TAN_POLY_3_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(TAN_POLY_2_D));
    res = _mm256_mlaf_pd(res, x2, _mm256_set1_pd(TAN_POLY_1_D));
    res = _mm256_mlaf_pd(res, _mm256_mul_pd(x2, r), r);

    res = _mm256_div_pd(
        _mm256_mul_pd(_mm256_set1_pd(2.0), res),
        _mm256_sub_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(res, res)),
    );

    res = _mm256_selecti_pd(is_even, res, _mm256_div_pd(_mm256_set1_pd(1.), res));
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::general::_mm256_extract_pd;

    #[test]
    fn test_tand() {
        unsafe {
            let value = _mm256_set1_pd(-2.70752239);
            let comparison = _mm256_tan_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = 0.46355663942902536f64;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm256_set1_pd(2.);
            let comparison = _mm256_tan_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = -2.185039863261519f64;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm256_set1_pd(-2.);
            let comparison = _mm256_tan_pd(value);
            let flag_1 = _mm256_extract_pd::<0>(comparison);
            let control = 2.185039863261519f64;
            assert_eq!(flag_1, control);
        }
    }
}
