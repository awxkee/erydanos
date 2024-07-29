/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::shuffle::_mm_shuffle;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
/// Compare *greater than or equal to* unsigned 8,
pub unsafe fn _mm_cmpge_epu8(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpeq_epi8(_mm_max_epu8(a, b), a)
}

#[inline(always)]
/// Compare *less than or equal to* unsigned 8,
pub unsafe fn _mm_cmple_epu8(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpge_epu8(b, a)
}

#[inline(always)]
/// Compare *greater than* unsigned 8,
pub unsafe fn _mm_cmpgt_epu8(a: __m128i, b: __m128i) -> __m128i {
    _mm_xor_si128(_mm_cmple_epu8(a, b), _mm_set1_epi8(-1))
}

#[inline(always)]
/// Compare *less than* unsigned 8,
pub unsafe fn _mm_cmplt_epu8(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epu8(b, a)
}

#[inline(always)]
/// Compare *greater than or equal to* unsigned 16,
pub unsafe fn _mm_cmpge_epu16(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpeq_epi16(_mm_max_epu16(a, b), a)
}

#[inline(always)]
/// Compare *less than or equal to* unsigned 16,
pub unsafe fn _mm_cmple_epu16(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpge_epu16(b, a)
}

#[inline(always)]
/// Compare *greater than* unsigned 16,
pub unsafe fn _mm_cmpgt_epu16(a: __m128i, b: __m128i) -> __m128i {
    _mm_xor_si128(_mm_cmple_epu16(a, b), _mm_set1_epi16(-1))
}

#[inline(always)]
/// Compare *less than* unsigned 16,
pub unsafe fn _mm_cmplt_epu16(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epu16(b, a)
}

#[inline(always)]
/// Compare *greater than or equal to* unsigned 32,
pub unsafe fn _mm_cmpge_epu32(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpeq_epi32(_mm_max_epu32(a, b), a)
}

#[inline(always)]
/// Compare *less than or equal to* unsigned 32,
pub unsafe fn _mm_cmple_epu32(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpge_epu16(b, a)
}

#[inline(always)]
/// Compare *greater than* unsigned 32,
pub unsafe fn _mm_cmpgt_epu32(a: __m128i, b: __m128i) -> __m128i {
    _mm_xor_si128(_mm_cmple_epu32(a, b), _mm_set1_epi32(-1))
}

#[inline(always)]
/// Compare *less than* unsigned 16,
pub unsafe fn _mm_cmplt_epu32(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epu32(b, a)
}

#[inline(always)]
/// Takes max for epi64
pub unsafe fn _mm_max_epi64x(a: __m128i, b: __m128i) -> __m128i {
    let mask = _mm_cmpgt_epi64(a, b);
    _mm_blendv_epi8(b, a, mask)
}

#[inline(always)]
/// Takes min for epi64
pub unsafe fn _mm_min_epi64x(a: __m128i, b: __m128i) -> __m128i {
    let mut mask = _mm_cmpgt_epi64(a, b);
    let ffs = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
    mask = _mm_andnot_si128(mask, ffs);
    _mm_blendv_epi8(b, a, mask)
}

#[inline(always)]
/// Compare *greater than or equal to* unsigned 64 bytes integers,
pub unsafe fn _mm_cmpge_epu64(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpeq_epi64(_mm_max_epi64x(a, b), a)
}

#[inline(always)]
/// Compare *less than or equal to* unsigned 64 bytes integers,
pub unsafe fn _mm_cmple_epu64(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpge_epu64(b, a)
}

#[inline(always)]
/// Compare *greater than* unsigned 64 bytes integers,
pub unsafe fn _mm_cmpgt_epu64(a: __m128i, b: __m128i) -> __m128i {
    let b = _mm_xor_si128(b, a);
    let a = _mm_and_si128(a, b);
    let b = _mm_sub_epi64(_mm_srli_epi64::<1>(b), a);
    const SHUFFLE_FLAG: i32 = _mm_shuffle(3, 3, 1, 1);
    _mm_shuffle_epi32::<SHUFFLE_FLAG>(_mm_srai_epi32::<31>(b))
}

#[inline(always)]
/// Compare *less than* unsigned 64 bytes integers,
pub unsafe fn _mm_cmplt_epu64(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epu64(b, a)
}

#[inline(always)]
/// Compare *greater than or equal to* signed 64 bytes integers,
pub unsafe fn _mm_cmpge_epi64(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpeq_epi64(_mm_max_epi64x(a, b), a)
}

#[inline(always)]
/// Compare *less than or equal to* signed 64 bytes integers,
pub unsafe fn _mm_cmple_epi64(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpge_epi64(b, a)
}

#[inline(always)]
/// Compare *less than* signed 64 bytes integers,
pub unsafe fn _mm_cmplt_epi64(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epi64(b, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::epi64::_mm_setr_epi64x;

    #[test]
    fn test_ge_epi64() {
        unsafe {
            let top = _mm_setr_epi64x(23, 0);
            let low = _mm_setr_epi64x(6, 15);
            let comparison = _mm_cmpgt_epi64(top, low);
            let flag_1 = _mm_extract_epi64::<0>(comparison);
            let flag_2 = _mm_extract_epi64::<1>(comparison);
            assert_ne!(flag_1, 0);
            assert_eq!(flag_2, 0);
        }

        unsafe {
            let top = _mm_setr_epi64x(23, -5);
            let low = _mm_setr_epi64x(-6, 15);
            let comparison = _mm_cmpgt_epi64(top, low);
            let flag_1 = _mm_extract_epi64::<0>(comparison);
            let flag_2 = _mm_extract_epi64::<1>(comparison);
            assert_ne!(flag_1, 0);
            assert_eq!(flag_2, 0);
        }
    }

    #[test]
    fn test_ge_epu64() {
        unsafe {
            let top = _mm_setr_epi64x(23, 0);
            let low = _mm_setr_epi64x(6, 15);
            let comparison = _mm_cmpgt_epu64(top, low);
            let flag_1 = _mm_extract_epi64::<0>(comparison);
            let flag_2 = _mm_extract_epi64::<1>(comparison);
            assert_ne!(flag_1, 0);
            assert_eq!(flag_2, 0);
        }

        unsafe {
            let top = _mm_setr_epi64x(23, -5);
            let low = _mm_setr_epi64x(-6, 15);
            let comparison = _mm_cmpgt_epu64(top, low);
            let flag_1 = _mm_extract_epi64::<0>(comparison);
            let flag_2 = _mm_extract_epi64::<1>(comparison);
            assert_eq!(flag_1, 0);
            assert_ne!(flag_2, 0);
        }
    }
}
