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
use crate::sse::general::_mm_rint_pd;
use crate::{_mm_cvtepi64_pd, _mm_mlaf_pd, _mm_mul_epi64, _mm_neg_pd, _mm_selecti_pd};

#[inline]
/// Computes cosine function with error bound *ULP 2.0*
pub unsafe fn _mm_cos_pd(d: __m128d) -> __m128d {
    let j = _mm_rint_pd(_mm_sub_pd(
        _mm_mul_pd(d, _mm_set1_pd(std::f64::consts::FRAC_1_PI)),
        _mm_set1_pd(0.5),
    ));

    let p = _mm_mul_epi64(j, _mm_set1_epi64x(2));
    let q = _mm_add_epi64(_mm_set1_epi64x(1), p);

    let qf = _mm_cvtepi64_pd(q);

    let mut r = _mm_mlaf_pd(qf, _mm_set1_pd(-PI_A2 * 0.5), d);
    r = _mm_mlaf_pd(qf, _mm_set1_pd(-PI_B2 * 0.5), r);

    let x2 = _mm_mul_pd(r, r);

    r = _mm_selecti_pd(
        _mm_cmpeq_epi64(_mm_and_si128(q, _mm_set1_epi64x(2)), _mm_setzero_si128()),
        _mm_neg_pd(r),
        r,
    );
    let mut res = _mm_set1_pd(SIN_POLY_10_D);
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(SIN_POLY_9_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(SIN_POLY_8_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(SIN_POLY_7_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(SIN_POLY_6_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(SIN_POLY_5_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(SIN_POLY_4_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(SIN_POLY_3_D));
    res = _mm_mlaf_pd(res, x2, _mm_set1_pd(SIN_POLY_2_D));
    res = _mm_mlaf_pd(res, _mm_mul_pd(x2, r), r);
    res
}

#[cfg(test)]
mod tests {
    use crate::_mm_extract_pd;

    use super::*;

    #[test]
    fn test_cosd() {
        unsafe {
            let value = _mm_set1_pd(-2.70752239);
            let comparison = _mm_cos_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            let control = -0.9072614508830367052983518f64;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value = _mm_set1_pd(2.);
            let comparison = _mm_cos_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            let control = -0.416146836547142386997568229500f64;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value = _mm_set1_pd(-2.);
            let comparison = _mm_cos_pd(value);
            let flag_1 = _mm_extract_pd::<0>(comparison);
            let control = -0.416146836547142386997568229500f64;
            assert_eq!(flag_1, control);
        }
    }
}
