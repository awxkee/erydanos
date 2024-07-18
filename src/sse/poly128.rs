/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::_mm_mul_epu64;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Type represents u128, in low u64 and high u64 part
/// Lower parts, and upper parts are interleaved
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct _mm128x2i(__m128i, __m128i);

#[inline]
/// Widening multiplication u64 in u128
pub unsafe fn _mm_mull_epu64(a: __m128i, b: __m128i) -> _mm128x2i {
    #[allow(overflowing_literals)]
    let upper_mask = _mm_set1_epi64x(0x_ffff_ffff_f800_0000);
    let xh = _mm_and_si128(a, upper_mask);
    let xl = _mm_sub_epi64(a, xh);
    let yh = _mm_and_si128(b, upper_mask);
    let yl = _mm_sub_epi64(b, yh);
    let r0 = _mm_mul_epu64(a, b);
    let d0 = r0;
    let j0 = _mm_mul_epu64(xh, yh);
    let j1 = _mm_mul_epu64(xl, yh);
    let j2 = _mm_mul_epu64(xh, yl);
    let j3 = _mm_mul_epu64(xl, yl);
    let d1 = _mm_sub_epi64(
        _mm_add_epi64(_mm_add_epi64(j0, j1), _mm_add_epi64(j2, j3)),
        r0,
    );
    _mm128x2i(d0, d1)
}

#[inline]
/// Shifts right u128 immediate
pub unsafe fn _mm_srli_epi128<const BITS: i32>(a: _mm128x2i) -> _mm128x2i {
    _mm128x2i(_mm_srli_epi64::<BITS>(a.0), _mm_srli_epi64::<BITS>(a.1))
}

#[inline]
/// Shifts left u128 immediate
pub unsafe fn _mm_slli_epi128<const BITS: i32>(a: _mm128x2i) -> _mm128x2i {
    _mm128x2i(_mm_slli_epi64::<BITS>(a.0), _mm_slli_epi64::<BITS>(a.1))
}

#[inline]
/// Widening add u64 to u128
pub unsafe fn _mm_add_epi128(a: _mm128x2i, b: __m128i) -> _mm128x2i {
    let r0 = _mm_add_epi64(a.0, b);
    let v = _mm_sub_epi64(r0, a.0);
    _mm128x2i(
        r0,
        _mm_add_epi64(
            _mm_sub_epi64(a.0, _mm_sub_epi64(r0, v)),
            _mm_add_epi64(_mm_sub_epi64(b, v), a.1),
        ),
    )
}

#[inline]
/// Saturates 128 bit integers presentation into 64 bits
pub unsafe fn _mm_movn_epi128(a: _mm128x2i) -> __m128i {
    _mm_add_epi64(a.0, a.1)
}
