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
use crate::sse::general::_mm_rint_pd;
use crate::tan::{
    TAN_POLY_1_D, TAN_POLY_2_D, TAN_POLY_3_D, TAN_POLY_4_D, TAN_POLY_5_D, TAN_POLY_6_D,
    TAN_POLY_7_D, TAN_POLY_8_D, TAN_POLY_9_D,
};
use crate::{_mm_cvtepi64_pd, _mm_mlaf_pd, _mm_neg_pd, _mm_selecti_pd};

#[inline]
#[target_feature(enable = "sse4.1")]
/// Computes tan function with error bound *ULP 1.5*
pub unsafe fn _mm_tan_pd(d: __m128d) -> __m128d {
    let q = _mm_rint_pd(_mm_mul_pd(d, _mm_set1_pd(std::f64::consts::FRAC_2_PI)));
    let qf = _mm_cvtepi64_pd(q);

    let mut r = _mm_mlaf_pd(qf, _mm_set1_pd(-PI_A2 * 0.5), d);
    r = _mm_mlaf_pd(qf, _mm_set1_pd(-PI_B2 * 0.5), r);

    let is_even = _mm_cmpeq_epi64(_mm_and_si128(q, _mm_set1_epi64x(1)), _mm_setzero_si128());

    r = _mm_selecti_pd(is_even, r, _mm_neg_pd(r));

    r = _mm_mul_pd(r, _mm_set1_pd(0.5));

    let x2 = _mm_mul_pd(r, r);

    let mut res = _mm_set1_pd(TAN_POLY_9_D);
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(TAN_POLY_8_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(TAN_POLY_7_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(TAN_POLY_6_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(TAN_POLY_5_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(TAN_POLY_4_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(TAN_POLY_3_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(TAN_POLY_2_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(TAN_POLY_1_D));
    res = _mm_mlaf_pd(res, _mm_mul_pd(x2, r), r);

    res = _mm_div_pd(
        _mm_mul_pd(_mm_set1_pd(2.0), res),
        _mm_sub_pd(_mm_set1_pd(1.0), _mm_mul_pd(res, res)),
    );

    res = _mm_selecti_pd(is_even, res, _mm_div_pd(_mm_set1_pd(1.), res));
    res
}

#[cfg(test)]
mod tests {
    use crate::_mm_extract_pd;

    use super::*;

    #[test]
    fn test_tanf() {
        unsafe {
            let value = _mm_set1_pd(-2.70752239);
            let comparison = _mm_tan_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            let control = 0.46355663942902536f64;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm_set1_pd(2.);
            let comparison = _mm_tan_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            let control = -2.185039863261519f64;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm_set1_pd(-2.);
            let comparison = _mm_tan_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            let control = 2.185039863261519f64;
            assert_eq!(flag_1, control);
        }
    }
}
