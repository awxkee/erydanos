/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::sse::epi64::_mm_blendv_epi64;
use crate::sse::unsigned::_mm_cmplt_epu32;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
/// Add unsigned 32 bytes integers using saturation
pub unsafe fn _mm_adds_epu64(a: __m128i, b: __m128i) -> __m128i {
    let result = _mm_add_epi64(a, b);
    let mask = _mm_cmplt_epu32(result, a);
    #[allow(overflowing_literals)]
    let saturated = _mm_set1_epi64x(0xFFFFFFFFFFFFFFFF);
    _mm_or_si128(
        _mm_and_si128(mask, saturated),
        _mm_andnot_si128(mask, result),
    )
}

#[inline(always)]
/// Add signed 64 bytes integers usign saturation
pub unsafe fn _mm_adds_epi64(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_add_epi64(lhs, rhs);

    _mm_blendv_epi64(
        res,
        _mm_blendv_epi64(_mm_set1_epi64x(i64::MIN), _mm_set1_epi64x(i64::MAX), res),
        _mm_xor_si128(rhs, _mm_cmpgt_epi64(lhs, res)),
    )
}

#[inline(always)]
/// subtract signed integers 64 using saturation
pub unsafe fn _mm_subs_epi64(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_sub_epi64(lhs, rhs);

    _mm_blendv_epi64(
        res,
        _mm_blendv_epi64(_mm_set1_epi64x(i64::MIN), _mm_set1_epi64x(i64::MAX), res),
        _mm_xor_si128(
            _mm_cmpgt_epi64(rhs, _mm_setzero_si128()),
            _mm_cmpgt_epi64(lhs, res),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_mm_extract_epi64x;

    #[test]
    fn saturation_add_epu64() {
        unsafe {
            let value1 = _mm_set1_epi64x(u64::MAX as i64);
            let value2 = _mm_set1_epi64x(15);

            let comparison = _mm_adds_epu64(value1, value2);
            let flag_1 = _mm_extract_epi64x::<0>(comparison) as u64;
            let control = u64::MAX;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value1 = _mm_set1_epi64x(15);
            let value2 = _mm_set1_epi64x(15);

            let comparison = _mm_adds_epu64(value1, value2);
            let flag_1 = _mm_extract_epi64x::<0>(comparison) as u64;
            let control = 30;
            assert_eq!(flag_1, control);
        }
    }
}
