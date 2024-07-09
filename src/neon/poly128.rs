/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::general::{vmul_u64, vmulq_u64};
use std::arch::aarch64::*;

/// Type represents u128, in low u64 and high u64 part
/// We'll keep this splat because mainly it is required to perform operations in both of them sequentially
///
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct u128x1_t(uint64x1_t, uint64x1_t);

/// Type represents u128, in low u64 and high u64 part
/// We'll keep this splat because mainly it is required to perform operations in both of them sequentially
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct s128x1_t(int64x1_t, int64x1_t);

/// Type represents u128, in low u64 and high u64 part
/// We'll keep this splat because mainly it is required to perform operations in both of them sequentially
///
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct u128x2_t(u128x1_t, u128x1_t);

/// Type represents i128, in low i64 and high i64 part
/// We'll keep this splat because mainly it is required to perform operations in both of them sequentially
///
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct s128x2_t(s128x1_t, s128x1_t);

/// Computes i128 as i64 and extracts in general register
#[inline(always)]
pub unsafe fn vextract_s128(d: s128x1_t) -> i64 {
    let product = vadd_s64(d.0, d.1);
    vget_lane_s64::<0>(product)
}

/// Computes u128 as u64 and extracts in general register
#[inline(always)]
pub unsafe fn vextract_u128(d: u128x1_t) -> u64 {
    let product = vadd_u64(d.0, d.1);
    vget_lane_u64::<0>(product)
}

/// Casts uint128 to int128
#[inline(always)]
pub unsafe fn vreinterpret_u128_s128(d: u128x1_t) -> s128x1_t {
    s128x1_t(vreinterpret_s64_u64(d.0), vreinterpret_s64_u64(d.1))
}

/// Broadcasts u64 into u128
#[inline(always)]
pub unsafe fn vmovl_u64(v: uint64x1_t) -> u128x1_t {
    u128x1_t(v, vdup_n_u64(0u64))
}

/// Broadcasts u64 into u128
#[inline(always)]
pub unsafe fn vmovl_s64(v: int64x1_t) -> s128x1_t {
    s128x1_t(v, vdup_n_s64(0i64))
}

/// Convert u64x2 low part as a low part of u128, and high as high of u128
#[inline(always)]
pub unsafe fn vcvtq_u128_u64(v: uint64x2_t) -> u128x1_t {
    u128x1_t(vget_low_u64(v), vget_high_u64(v))
}

#[inline(always)]
pub unsafe fn vmovnq_u128(p: u128x2_t) -> uint64x2_t {
    let v0 = vadd_u64(p.0 .0, p.0 .1);
    let v1 = vadd_u64(p.1 .0, p.1 .1);
    vcombine_u64(v0, v1)
}

#[inline(always)]
pub unsafe fn vmovn_u128(p: u128x1_t) -> uint64x1_t {
    let v0 = vadd_u64(p.0, p.1);
    v0
}

/// Convert u64x2 low part as a low part of u128, and high as high of u128
#[inline(always)]
pub unsafe fn vcreate_u128(low: uint64x1_t, high: uint64x1_t) -> u128x1_t {
    u128x1_t(low, high)
}

#[inline]
/// Widening multiplication u64 in u128
pub unsafe fn vmull_s64(a: int64x1_t, b: int64x1_t) -> s128x1_t {
    let sign_ab = vshr_n_u64::<63>(vreinterpret_u64_s64(a));
    let sign_cd = vshr_n_u64::<63>(vreinterpret_u64_s64(b));
    let sign = veor_u64(sign_ab, sign_cd);
    let uab = vreinterpret_u64_s64(vabs_s64(a));
    let ucd = vreinterpret_u64_s64(vabs_s64(b));
    let product = vmull_u64(uab, ucd);
    let signed_product = vreinterpret_u128_s128(product);
    let integral_part = vbsl_s64(
        vceqz_u64(sign),
        signed_product.0,
        vneg_s64(signed_product.0),
    );
    s128x1_t(integral_part, signed_product.1)
}

#[inline]
/// Widening multiplication u64 in u128
pub unsafe fn vmull_u64(a: uint64x1_t, b: uint64x1_t) -> u128x1_t {
    let xh = vshr_n_u64::<32>(a);
    let xl = vsub_u64(a, xh);
    let yh = vshr_n_u64::<32>(b);
    let yl = vsub_u64(b, yh);
    let r0 = vmul_u64(a, b);
    let d0 = r0;
    let j0 = vmul_u64(xh, yh);
    let j1 = vmul_u64(xl, yh);
    let j2 = vmul_u64(xh, yl);
    let j3 = vmul_u64(xl, yl);
    let d1 = vsub_u64(vadd_u64(vadd_u64(j0, j1), vadd_u64(j2, j3)), r0);
    u128x1_t(d0, d1)
}

#[inline]
/// Widening multiplication u64 in u128
pub unsafe fn vmullq_u64(a: uint64x2_t, b: uint64x2_t) -> u128x2_t {
    let upper_mask = vdupq_n_u64(0x_ffff_ffff_f800_0000);
    let xh = vandq_u64(a, upper_mask);
    let xl = vsubq_u64(a, xh);
    let yh = vandq_u64(b, upper_mask);
    let yl = vsubq_u64(b, yh);
    let r0 = vmulq_u64(a, b);
    let d0 = r0;
    let j0 = vmulq_u64(xh, yh);
    let j1 = vmulq_u64(xl, yh);
    let j2 = vmulq_u64(xh, yl);
    let j3 = vmulq_u64(xl, yl);
    let d1 = vsubq_u64(vaddq_u64(vaddq_u64(j0, j1), vaddq_u64(j2, j3)), r0);
    u128x2_t(
        vcreate_u128(vget_low_u64(d0), vget_low_u64(d1)),
        vcreate_u128(vget_high_u64(d0), vget_high_u64(d1)),
    )
}

#[inline]
/// Widening add u64 to u128
pub unsafe fn vaddw_u64(a: u128x1_t, b: uint64x1_t) -> u128x1_t {
    let r0 = vadd_u64(a.0, b);
    let v = vsub_u64(r0, a.0);
    u128x1_t(
        r0,
        vadd_u64(
            vsub_u64(a.0, vsub_u64(r0, v)),
            vadd_u64(vsub_u64(b, v), a.1),
        ),
    )
}

#[inline]
/// Widening add i64 to i128
pub unsafe fn vaddw_s64(a: s128x1_t, b: int64x1_t) -> s128x1_t {
    let r0 = vadd_s64(a.0, b);
    let v = vsub_s64(r0, a.0);
    s128x1_t(
        r0,
        vadd_s64(
            vsub_s64(a.0, vsub_s64(r0, v)),
            vadd_s64(vsub_s64(b, v), a.1),
        ),
    )
}

#[inline]
/// Widening add u64 to u128
pub unsafe fn vaddwq_u64(a: u128x2_t, b: uint64x2_t) -> u128x2_t {
    let r0 = vaddw_u64(a.0, vget_low_u64(b));
    let r1 = vaddw_u64(a.1, vget_high_u64(b));
    u128x2_t(r0, r1)
}

#[inline]
/// Widening add i64 to i128
pub unsafe fn vaddwq_s64(a: s128x2_t, b: int64x2_t) -> s128x2_t {
    let r0 = vaddw_s64(a.0, vget_low_s64(b));
    let r1 = vaddw_s64(a.1, vget_high_s64(b));
    s128x2_t(r0, r1)
}

#[inline]
/// Shifts right u128 immediate
pub unsafe fn vshr_n_u128<const BITS: i32>(a: u128x1_t) -> u128x1_t {
    u128x1_t(vshr_n_u64::<BITS>(a.0), vshr_n_u64::<BITS>(a.1))
}

#[inline]
/// Shifts right u128 immediate
pub unsafe fn vshrq_n_u128<const BITS: i32>(a: u128x2_t) -> u128x2_t {
    u128x2_t(vshr_n_u128::<BITS>(a.0), vshr_n_u128::<BITS>(a.1))
}

#[inline]
/// Shifts right s128 immediate
pub unsafe fn vshr_n_s128<const BITS: i32>(a: s128x1_t) -> s128x1_t {
    s128x1_t(vshr_n_s64::<BITS>(a.0), vshr_n_s64::<BITS>(a.1))
}

#[inline]
/// Shifts right s128 immediate
pub unsafe fn vshrq_n_s128<const BITS: i32>(a: s128x2_t) -> s128x2_t {
    s128x2_t(vshr_n_s128::<BITS>(a.0), vshr_n_s128::<BITS>(a.1))
}
