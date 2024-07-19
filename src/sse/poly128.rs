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

use crate::sse::epi64::_mm_not_epi64;
use crate::{
    _mm_abs_epi64, _mm_cmplt_epi64, _mm_cmplt_epu64, _mm_extract_epi64x, _mm_neg_epi64,
    _mm_packus_epi64, _mm_select_epi64, _mm_sllv_epi64x,
};

/// Type represents u128, in low u64 and high u64 part
/// Lower parts, and upper parts are interleaved
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct __m128x2i(__m128i, __m128i);

#[inline]
/// Widening multiplication u64 in u128
pub unsafe fn _mm_mull_epu64(a: __m128i, b: __m128i) -> __m128x2i {
    let erase_high = _mm_set1_epi64x(0xFFFFFFFF);
    let xh = _mm_srli_epi64::<32>(a);
    let xl = _mm_and_si128(a, erase_high);
    let yh = _mm_srli_epi64::<32>(b);
    let yl = _mm_and_si128(b, erase_high);

    let xh = _mm_packus_epi64(xh, xh);
    let xl = _mm_packus_epi64(xl, xl);
    let yh = _mm_packus_epi64(yh, yh);
    let yl = _mm_packus_epi64(yl, yl);

    let hh = _mm_mul_epu32(xh, yh);
    let lh = _mm_mul_epu32(xl, yh);
    let hl = _mm_mul_epu32(xh, yl);
    let ll = _mm_mul_epu32(xl, yl);

    let lo = ll;
    let hi = hh;
    let mut rs = __m128x2i(lo, hi);

    let carry1 = _mm_slli_epi128x::<32>(__m128x2i(lh, _mm_set1_epi64x(0)));
    let carry2 = _mm_slli_epi128x::<32>(__m128x2i(hl, _mm_set1_epi64x(0)));

    rs = _mm_add_epu128(rs, carry1);
    rs = _mm_add_epu128(rs, carry2);
    rs
}

#[inline]
/// Widening multiplication i64 in i128
pub unsafe fn _mm_mull_epi64(a: __m128i, b: __m128i) -> __m128x2i {
    let sign_ab = _mm_srli_epi64::<63>(a);
    let sign_cd = _mm_srli_epi64::<63>(b);
    let sign = _mm_xor_si128(sign_ab, sign_cd);
    let uab = _mm_abs_epi64(a);
    let ucd = _mm_abs_epi64(b);
    let product = _mm_mull_epu64(uab, ucd);
    let is_not_negative = _mm_cmpeq_epi64(sign, _mm_setzero_si128());
    let lo = _mm_select_epi64(is_not_negative, product.0, _mm_not_epi64(product.0));
    let hi = _mm_select_epi64(is_not_negative, product.1, _mm_not_epi64(product.1));
    let v0 = _mm_select_epi64(is_not_negative, lo, _mm_add_epi64(lo, _mm_set1_epi64x(1)));
    let v1 = hi;
    __m128x2i(v0, v1)
}

#[inline]
/// Shifts right u128 immediate
pub unsafe fn _mm_srli_epi128x<const IMM: i32>(a: __m128x2i) -> __m128x2i {
    return if IMM <= 0 {
        a
    } else if IMM < 64 {
        let upper_shift = _mm_set1_epi64x(64i64 - IMM as i64);
        let new_lower = _mm_or_si128(
            _mm_sllv_epi64x(a.1, upper_shift),
            _mm_srli_epi64::<IMM>(a.0),
        );
        let new_upper = _mm_srli_epi64::<IMM>(a.1);
        __m128x2i(new_lower, new_upper)
    } else {
        let shr_value = _mm_set1_epi64x(64i64 - IMM as i64);
        __m128x2i(_mm_sllv_epi64x(a.1, shr_value), _mm_set1_epi64x(0))
    };
}

#[inline]
/// Shifts left u128 immediate
pub unsafe fn _mm_slli_epi128x<const IMM: i32>(a: __m128x2i) -> __m128x2i {
    if IMM >= 64 {
        let v_imm = _mm_set1_epi64x(IMM as i64 - 64i64);
        __m128x2i(_mm_set1_epi64x(0), _mm_sll_epi64(a.1, v_imm))
    } else {
        let v_imm = _mm_set1_epi64x(IMM as i64);
        let lo = _mm_sll_epi64(a.0, v_imm);
        let mut hi = _mm_sllv_epi64x(a.1, v_imm);
        let overflow = _mm_srl_epi64(a.0, _mm_set1_epi64x(64i64 - IMM as i64));
        hi = _mm_or_si128(hi, overflow);
        __m128x2i(lo, hi)
    }
}

#[inline]
/// Widening add 64 bytes integer to 128 bytes integer
pub unsafe fn _mm_addw_epi128(a: __m128x2i, b: __m128i) -> __m128x2i {
    let r0 = _mm_add_epi64(a.0, b);
    let v = _mm_sub_epi64(r0, a.0);
    __m128x2i(
        r0,
        _mm_add_epi64(
            _mm_sub_epi64(a.0, _mm_sub_epi64(r0, v)),
            _mm_add_epi64(_mm_sub_epi64(b, v), a.1),
        ),
    )
}

#[inline]
/// Widening substract 64 bytes integer to 128 bytes integer
pub unsafe fn _mm_subw_epi128(a: __m128x2i, b: __m128i) -> __m128x2i {
    _mm_addw_epi128(a, _mm_neg_epi64(b))
}

#[inline]
/// Saturates 128-bit integers presentation into 64 bits
pub unsafe fn _mm_movn_epi128(a: __m128x2i) -> __m128i {
    _mm_add_epi64(a.0, a.1)
}

#[inline]
/// Takes absolute value for i128
pub unsafe fn _mm_abs_epi128(a: __m128x2i) -> __m128x2i {
    let is_neg = _mm_cmplt_epi64(a.1, _mm_setzero_si128());
    __m128x2i(
        _mm_select_epi64(is_neg, _mm_neg_epi64(a.0), a.0),
        _mm_select_epi64(is_neg, _mm_neg_epi64(a.1), a.1),
    )
}

/// Computes i128 as u64 and extracts lower half in general register
#[inline(always)]
pub unsafe fn _mm_extract_lo_epi128<const IMM: i32>(d: __m128x2i) -> i64 {
    _mm_extract_epi64x::<IMM>(d.0)
}

/// Computes i128 as u64 and extracts upper half in general register
#[inline(always)]
pub unsafe fn _mm_extract_hi_epi128<const IMM: i32>(d: __m128x2i) -> i64 {
    _mm_extract_epi64x::<IMM>(d.1)
}

/// Computes u128 as u128 and extracts in general register
#[inline(always)]
pub unsafe fn _mm_extract_epu128<const IMM: i32>(d: __m128x2i) -> u128 {
    let lo = (_mm_extract_epi64x::<IMM>(d.0) as u64) as u128;
    let hi = (_mm_extract_epi64x::<IMM>(d.1) as u64) as u128;
    lo | (hi << 64)
}

/// Computes u128 as u128 and extracts in general register
#[inline(always)]
pub unsafe fn _mm_extract_epi128<const IMM: i32>(d: __m128x2i) -> i128 {
    let lo = (_mm_extract_epi64x::<IMM>(d.0) as u64) as u128;
    let hi = (_mm_extract_epi64x::<IMM>(d.1) as u64) as u128;
    (lo | (hi << 64)) as i128
}

#[inline(always)]
/// Adds s128 to s128 using signed addition
pub unsafe fn _mm_add_epi128(a: __m128x2i, b: __m128x2i) -> __m128x2i {
    let lo = _mm_add_epi64(a.0, b.0);

    let overflow = _mm_cmplt_epi64(
        _mm_and_si128(
            _mm_not_epi64(_mm_xor_si128(a.0, b.0)),
            _mm_xor_si128(a.0, lo),
        ),
        _mm_setzero_si128(),
    );

    let mut carry = _mm_select_epi64(overflow, _mm_set1_epi64x(1), _mm_set1_epi64x(0));
    carry = _mm_select_epi64(
        _mm_cmplt_epi64(b.1, _mm_setzero_si128()),
        _mm_set1_epi64x(-1),
        carry,
    );
    __m128x2i(lo, _mm_add_epi64(_mm_add_epi64(a.1, b.1), carry))
}

#[inline(always)]
/// Adds s128 to s128 using unsigned addition
pub unsafe fn _mm_add_epu128(a: __m128x2i, b: __m128x2i) -> __m128x2i {
    let lo = _mm_add_epi64(a.0, b.0);
    let carry = _mm_select_epi64(
        _mm_cmplt_epu64(lo, a.0),
        _mm_set1_epi64x(1),
        _mm_set1_epi64x(0),
    );
    __m128x2i(lo, _mm_add_epi64(_mm_add_epi64(a.1, b.1), carry))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        unsafe {
            let lhs = __m128x2i(_mm_set1_epi64x(27), _mm_set1_epi64x(0));
            let rhs = __m128x2i(_mm_set1_epi64x(54), _mm_set1_epi64x(0));
            let comparison = _mm_add_epi128(lhs, rhs);
            let flag_1 = _mm_extract_epi128::<0>(comparison);
            let control = 27 + 54;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let lhs = __m128x2i(_mm_set1_epi64x(27), _mm_set1_epi64x(1));
            let rhs = __m128x2i(_mm_set1_epi64x(54), _mm_set1_epi64x(1));
            let comparison = _mm_add_epi128(lhs, rhs);
            let flag_1 = _mm_extract_epi128::<0>(comparison);
            let control = (27i128 | (1i128 << 64)) + (54i128 | (1i128 << 64));
            assert_eq!(flag_1, control);
        }

        unsafe {
            let lhs = __m128x2i(_mm_set1_epi64x(27), _mm_set1_epi64x(0));
            let rhs = __m128x2i(_mm_set1_epi64x(54), _mm_set1_epi64x(0));
            let comparison = _mm_add_epu128(lhs, rhs);
            let flag_1 = _mm_extract_epu128::<0>(comparison);
            let control = 27 + 54;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let lhs = __m128x2i(_mm_set1_epi64x(27), _mm_set1_epi64x(1));
            let rhs = __m128x2i(_mm_set1_epi64x(54), _mm_set1_epi64x(1));
            let comparison = _mm_add_epu128(lhs, rhs);
            let flag_1 = _mm_extract_epu128::<0>(comparison);
            let control = (27u128 | (1u128 << 64)) + (54u128 | (1u128 << 64));
            assert_eq!(flag_1, control);
        }
    }

    #[test]
    fn test_srli() {
        unsafe {
            let comparison =
                _mm_srli_epi128x::<2>(__m128x2i(_mm_set1_epi64x(27), _mm_set1_epi64x(0)));
            let flag_1 = _mm_extract_epi128::<0>(comparison);
            let control = 27 >> 2;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let comparison =
                _mm_srli_epi128x::<2>(__m128x2i(_mm_set1_epi64x(27), _mm_set1_epi64x(1)));
            let flag_1 = _mm_extract_epi128::<0>(comparison);
            let control = (27i128 | (1i128 << 64i128)) >> 2;
            assert_eq!(flag_1, control);
        }
    }

    #[test]
    fn test_slli() {
        unsafe {
            let comparison =
                _mm_slli_epi128x::<2>(__m128x2i(_mm_set1_epi64x(27), _mm_set1_epi64x(0)));
            let flag_1 = _mm_extract_epi128::<0>(comparison);
            let control = 27 << 2;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let comparison =
                _mm_slli_epi128x::<32>(__m128x2i(_mm_set1_epi64x(27), _mm_set1_epi64x(0)));
            let flag_1 = _mm_extract_epi128::<0>(comparison);
            let control = 27 << 32;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let comparison =
                _mm_slli_epi128x::<32>(__m128x2i(_mm_set1_epi64x(10505000), _mm_set1_epi64x(0)));
            let flag_1 = _mm_extract_epi128::<0>(comparison);
            let control = 10505000 << 32;
            assert_eq!(flag_1, control);
        }
    }

    #[test]
    fn test_widen_mul() {
        unsafe {
            let value1 = _mm_set1_epi64x(27);
            let value2 = _mm_set1_epi64x(2);

            let comparison = _mm_mull_epu64(value1, value2);
            let flag_1 = _mm_extract_lo_epi128::<0>(comparison);
            let control = 27 * 2;
            assert_eq!(flag_1, control);
        }

        unsafe {
            #[allow(overflowing_literals)]
            let value1 = _mm_set1_epi64x(u64::MAX as i64);
            let value2 = _mm_set1_epi64x(2);

            let comparison = _mm_mull_epu64(value1, value2);
            let flag_1 = _mm_extract_epu128::<0>(comparison);
            let control = u64::MAX as u128 * 2;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value1 = _mm_set1_epi64x(27);
            let value2 = _mm_set1_epi64x(2);

            let comparison = _mm_mull_epi64(value1, value2);
            let flag_1 = _mm_extract_lo_epi128::<0>(comparison);
            let control = 27 * 2;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value1 = _mm_set1_epi64x(-27);
            let value2 = _mm_set1_epi64x(2);

            let comparison = _mm_mull_epi64(value1, value2);
            let flag_1 = _mm_extract_lo_epi128::<0>(comparison);
            let control = -27 * 2;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value1 = _mm_set1_epi64x(27);
            let value2 = _mm_set1_epi64x(-2);

            let comparison = _mm_mull_epi64(value1, value2);
            let flag_1 = _mm_extract_lo_epi128::<0>(comparison);
            let control = -27 * 2;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value1 = _mm_set1_epi64x(i64::MAX);
            let value2 = _mm_set1_epi64x(-2);

            let comparison = _mm_mull_epi64(value1, value2);
            let flag_1 = _mm_extract_epi128::<0>(comparison);
            let control = -i64::MAX as i128 * 2;
            assert_eq!(flag_1, control);
        }
    }
}
