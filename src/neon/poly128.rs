/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

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
/// Low parts and half parts are interleaved
///
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct u128x2_t(uint64x2_t, uint64x2_t);

/// Type represents i128, in low i64 and high i64 part
/// Low parts and half parts are interleaved
///
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct s128x2_t(int64x2_t, int64x2_t);

/// Computes i128 as i64 and extracts in general register
#[inline]
pub unsafe fn vextract_s128(d: s128x1_t) -> i64 {
    let product = vadd_s64(d.0, d.1);
    vget_lane_s64::<0>(product)
}

/// Computes u128 as u64 and extracts in general register
#[inline]
pub unsafe fn vextract_u128(d: u128x1_t) -> u64 {
    let product = vadd_u64(d.0, d.1);
    vget_lane_u64::<0>(product)
}

/// Computes i128 as i64 and extracts in general register
#[inline]
pub unsafe fn vextractq_lo_s128<const IMM: i32>(d: s128x2_t) -> i64 {
    vgetq_lane_s64::<IMM>(d.0)
}

/// Computes i128 as u64 and extracts in general register
#[inline]
pub unsafe fn vextractq_lo_u128<const IMM: i32>(d: u128x2_t) -> u64 {
    vgetq_lane_u64::<IMM>(d.0)
}

#[inline]
/// Computes i128 as u128 and extracts in general register
pub unsafe fn vextractq_q_u128<const IMM: i32>(d: u128x2_t) -> u128 {
    let v0 = vgetq_lane_u64::<IMM>(d.0) as u128;
    let v1 = vgetq_lane_u64::<IMM>(d.1) as u128;
    v0 | (v1 << 64)
}

#[inline]
/// Computes i128 as u128 and extracts in general register
pub unsafe fn vextractq_q_s128<const IMM: i32>(d: s128x2_t) -> i128 {
    let v0 = vgetq_lane_s64::<IMM>(d.0) as i128;
    let v1 = vgetq_lane_s64::<IMM>(d.1) as i128;
    v0 | (v1 << 64)
}

/// Casts uint128 to int128
#[inline]
pub unsafe fn vreinterpret_u128_s128(d: u128x1_t) -> s128x1_t {
    s128x1_t(vreinterpret_s64_u64(d.0), vreinterpret_s64_u64(d.1))
}

#[inline]
/// Casts uint128 to int128
pub unsafe fn vreinterpretq_u128_s128(d: u128x2_t) -> s128x2_t {
    s128x2_t(vreinterpretq_s64_u64(d.0), vreinterpretq_s64_u64(d.1))
}

/// Broadcasts u64 into u128
#[inline]
pub unsafe fn vmovl_u64(v: uint64x1_t) -> u128x1_t {
    u128x1_t(v, vdup_n_u64(0u64))
}

/// Broadcasts u64 into u128
#[inline]
pub unsafe fn vmovl_s64(v: int64x1_t) -> s128x1_t {
    s128x1_t(v, vdup_n_s64(0i64))
}

/// Convert u64x2 low part as a low part of u128, and high as high of u128
#[inline]
pub unsafe fn vcvtq_u128_u64(v: uint64x2_t) -> u128x1_t {
    u128x1_t(vget_low_u64(v), vget_high_u64(v))
}

#[inline]
pub unsafe fn vmovnq_u128(v: u128x2_t) -> uint64x2_t {
    vaddq_u64(v.0, v.1)
}

#[inline]
pub unsafe fn vmovn_u128(p: u128x1_t) -> uint64x1_t {
    let v0 = vadd_u64(p.0, p.1);
    v0
}

/// Convert u64x2 low part as a low part of u128, and high as high of u128
#[inline]
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
    let is_not_negative = vceqz_u64(sign);
    let lo = vbsl_u64(
        is_not_negative,
        product.0,
        vreinterpret_u64_u32(vmvn_u32(vreinterpret_u32_u64(product.0))),
    );
    let hi = vbsl_u64(
        is_not_negative,
        product.1,
        vreinterpret_u64_u32(vmvn_u32(vreinterpret_u32_u64(product.1))),
    );
    let v0 = vbsl_s64(
        is_not_negative,
        vreinterpret_s64_u64(lo),
        vadd_s64(vreinterpret_s64_u64(lo), vdup_n_s64(1)),
    );
    let v1 = vreinterpret_s64_u64(hi);
    s128x1_t(v0, v1)
}

#[inline]
/// Widening multiplication u64 in u128
pub unsafe fn vmullq_s64(a: int64x2_t, b: int64x2_t) -> s128x2_t {
    let sign_ab = vshrq_n_u64::<63>(vreinterpretq_u64_s64(a));
    let sign_cd = vshrq_n_u64::<63>(vreinterpretq_u64_s64(b));
    let sign = veorq_u64(sign_ab, sign_cd);
    let uab = vreinterpretq_u64_s64(vabsq_s64(a));
    let ucd = vreinterpretq_u64_s64(vabsq_s64(b));
    let product = vmullq_u64(uab, ucd);
    let is_not_negative = vceqzq_u64(sign);
    let lo = vbslq_u64(
        is_not_negative,
        product.0,
        vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(product.0))),
    );
    let hi = vbslq_u64(
        is_not_negative,
        product.1,
        vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(product.1))),
    );
    let v0 = vbslq_s64(
        is_not_negative,
        vreinterpretq_s64_u64(lo),
        vaddq_s64(vreinterpretq_s64_u64(lo), vdupq_n_s64(1)),
    );
    let v1 = vreinterpretq_s64_u64(hi);
    s128x2_t(v0, v1)
}

#[inline]
/// Widening multiplication u64 in u128
pub unsafe fn vmull_u64(a: uint64x1_t, b: uint64x1_t) -> u128x1_t {
    let ah = vcombine_u64(a, a);
    let bh = vcombine_u64(b, b);
    let product = vmullq_u64(ah, bh);
    u128x1_t(vget_low_u64(product.0), vget_high_u64(product.1))
}

#[inline]
/// Widening multiplication u64 in u128
pub unsafe fn vmullq_u64(a: uint64x2_t, b: uint64x2_t) -> u128x2_t {
    let erase_high = vdupq_n_u64(0xFFFFFFFF);
    let xh = vmovn_u64(vshrq_n_u64::<32>(a));
    let xl = vmovn_u64(vandq_u64(a, erase_high));
    let yh = vmovn_u64(vshrq_n_u64::<32>(b));
    let yl = vmovn_u64(vandq_u64(b, erase_high));

    let hh = vmull_u32(xh, yh);
    let lh = vmull_u32(xl, yh);
    let hl = vmull_u32(xh, yl);
    let ll = vmull_u32(xl, yl);

    let lo = ll;
    let hi = hh;
    let mut rs = u128x2_t(lo, hi);

    let carry1 = vshlq_n_u128::<32>(u128x2_t(lh, vdupq_n_u64(0)));
    let carry2 = vshlq_n_u128::<32>(u128x2_t(hl, vdupq_n_u64(0)));

    rs = vaddq_u128(rs, carry1);
    rs = vaddq_u128(rs, carry2);
    rs
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
    let j0 = vadd_s64(a.0, b);
    //   let overflow = ((!(a ^ b) & (a ^ sum)) < 0);
    let overflow = vcltz_s64(vand_s64(
        vreinterpret_s64_s32(vmvn_s32(vreinterpret_s32_s64(veor_s64(a.0, b)))),
        veor_s64(a.0, j0),
    ));
    let mut carry = vdup_n_s64(1);
    carry = vbsl_s64(vcltz_s64(b), vdup_n_s64(-1), carry);
    let hi = vbsl_s64(overflow, vadd_s64(a.1, carry), a.1);
    s128x1_t(j0, hi)
}

#[inline]

/// Widening add u64 to u128
pub unsafe fn vaddwq_u64(a: u128x2_t, b: uint64x2_t) -> u128x2_t {
    let j0 = vaddq_u64(a.0, b);
    let overflow = vcltq_u64(j0, a.0);
    let hi = vbslq_u64(overflow, vaddq_u64(a.1, vdupq_n_u64(1)), a.1);
    u128x2_t(j0, hi)
}

#[inline]
/// Adds u128 to u128
pub unsafe fn vaddq_u128(a: u128x2_t, b: u128x2_t) -> u128x2_t {
    let lo = vaddq_u64(a.0, b.0);
    let carry = vbslq_u64(vcltq_u64(lo, a.0), vdupq_n_u64(1), vdupq_n_u64(0));
    u128x2_t(lo, vaddq_u64(vaddq_u64(a.1, b.1), carry))
}

#[inline]
/// Adds s128 to s128
pub unsafe fn vaddq_s128(a: s128x2_t, b: s128x2_t) -> s128x2_t {
    let lo = vaddq_s64(a.0, b.0);

    let overflow = vcltzq_s64(vandq_s64(
        vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(veorq_s64(a.0, b.0)))),
        veorq_s64(a.0, lo),
    ));

    let mut carry = vbslq_s64(overflow, vdupq_n_s64(1), vdupq_n_s64(0));
    carry = vbslq_s64(vcltzq_s64(b.1), vdupq_n_s64(-1), carry);
    s128x2_t(lo, vaddq_s64(vaddq_s64(a.1, b.1), carry))
}

#[inline]
/// Widening add i64 to i128
pub unsafe fn vaddwq_s64(a: s128x2_t, b: int64x2_t) -> s128x2_t {
    let j0 = vaddq_s64(a.0, b);
    //   let overflow = ((!(a ^ b) & (a ^ sum)) < 0);
    let overflow = vcltzq_s64(vandq_s64(
        vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(veorq_s64(a.0, b)))),
        veorq_s64(a.0, j0),
    ));
    let mut carry = vdupq_n_s64(1);
    carry = vbslq_s64(vcltzq_s64(b), vdupq_n_s64(-1), carry);
    let hi = vbslq_s64(overflow, vaddq_s64(a.1, carry), a.1);
    s128x2_t(j0, hi)
}

#[inline]
/// Shifts right u128 immediate
pub unsafe fn vshr_n_u128<const IMM: i32>(a: u128x1_t) -> u128x1_t {
    return if IMM <= 0 {
        a
    } else if IMM < 64 {
        let upper_shift = vdup_n_s64(64i64 - IMM as i64);
        let new_lower = vorr_u64(vshl_u64(a.1, upper_shift), vshr_n_u64::<IMM>(a.0));
        let new_upper = vshr_n_u64::<IMM>(a.1);
        u128x1_t(new_lower, new_upper)
    } else {
        let shr_value = vdup_n_s64(-(64i64 - IMM as i64));
        u128x1_t(vshl_u64(a.1, shr_value), vdup_n_u64(0))
    };
}

#[inline]
/// Shifts right u128 immediate
pub unsafe fn vshrq_n_u128<const IMM: i32>(a: u128x2_t) -> u128x2_t {
    return if IMM <= 0 {
        a
    } else if IMM < 64 {
        let upper_shift = vdupq_n_s64(64i64 - IMM as i64);
        let new_lower = vorrq_u64(vshlq_u64(a.1, upper_shift), vshrq_n_u64::<IMM>(a.0));
        let new_upper = vshrq_n_u64::<IMM>(a.1);
        u128x2_t(new_lower, new_upper)
    } else {
        let shr_value = vdupq_n_s64(-(64i64 - IMM as i64));
        u128x2_t(vshlq_u64(a.1, shr_value), vdupq_n_u64(0))
    };
}

#[inline]
/// Shifts right s128 immediate
pub unsafe fn vshr_n_s128<const IMM: i32>(a: s128x1_t) -> s128x1_t {
    return if IMM <= 0 {
        a
    } else if IMM < 64 {
        let upper_shift = vdup_n_s64(64i64 - IMM as i64);
        let new_lower = vorr_s64(vshl_s64(a.1, upper_shift), vshr_n_s64::<IMM>(a.0));
        let new_upper = vshr_n_s64::<IMM>(a.1);
        s128x1_t(new_lower, new_upper)
    } else {
        let shr_value = vdup_n_s64(-(64i64 - IMM as i64));
        s128x1_t(vshl_s64(a.1, shr_value), vdup_n_s64(0))
    };
}

#[inline]
/// Shifts right s128 immediate
pub unsafe fn vshrq_n_s128<const IMM: i32>(a: s128x2_t) -> s128x2_t {
    return if IMM <= 0 {
        a
    } else if IMM < 64 {
        let upper_shift = vdupq_n_s64(64i64 - IMM as i64);
        let new_lower = vorrq_s64(vshlq_s64(a.1, upper_shift), vshrq_n_s64::<IMM>(a.0));
        let new_upper = vshrq_n_s64::<IMM>(a.1);
        s128x2_t(new_lower, new_upper)
    } else {
        let shr_value = vdupq_n_s64(-(64i64 - IMM as i64));
        s128x2_t(vshlq_s64(a.1, shr_value), vdupq_n_s64(0))
    };
}

#[inline]
/// Shifts left u128 immediate
pub unsafe fn vshl_n_u128<const IMM: i32>(a: u128x1_t) -> u128x1_t {
    if IMM >= 64 {
        let v_imm = vdup_n_s64(IMM as i64 - 64i64);
        u128x1_t(vdup_n_u64(0), vshl_u64(a.1, v_imm))
    } else {
        let v_imm = vdup_n_s64(IMM as i64);
        let lo = vshl_u64(a.0, v_imm);
        let mut hi = vshl_u64(a.1, v_imm);
        let overflow = vshl_u64(a.0, vdup_n_s64(-(64i64 - IMM as i64)));
        hi = vorr_u64(hi, overflow);
        u128x1_t(lo, hi)
    }
}

#[inline]
/// Shifts left u128 immediate
pub unsafe fn vshlq_n_u128<const IMM: i32>(a: u128x2_t) -> u128x2_t {
    if IMM >= 64 {
        let v_imm = vdupq_n_s64(IMM as i64 - 64i64);
        u128x2_t(vdupq_n_u64(0), vshlq_u64(a.1, v_imm))
    } else {
        let v_imm = vdupq_n_s64(IMM as i64);
        let lo = vshlq_u64(a.0, v_imm);
        let mut hi = vshlq_u64(a.1, v_imm);
        let overflow = vshlq_u64(a.0, vdupq_n_s64(-(64i64 - IMM as i64)));
        hi = vorrq_u64(hi, overflow);
        u128x2_t(lo, hi)
    }
}

#[inline]

/// Shifts left s128 immediate
pub unsafe fn vshl_n_s128<const IMM: i32>(a: s128x1_t) -> s128x1_t {
    if IMM >= 64 {
        let v_imm = vdup_n_s64(IMM as i64 - 64i64);
        s128x1_t(vdup_n_s64(0), vshl_s64(a.1, v_imm))
    } else {
        let v_imm = vdup_n_s64(IMM as i64);
        let lo = vshl_s64(a.0, v_imm);
        let mut hi = vshl_s64(a.1, v_imm);
        let overflow = vshl_s64(a.0, vdup_n_s64(-(64i64 - IMM as i64)));
        hi = vorr_s64(hi, overflow);
        s128x1_t(lo, hi)
    }
}

#[inline]
/// Shifts right s128 immediate
pub unsafe fn vshlq_n_s128<const IMM: i32>(a: s128x2_t) -> s128x2_t {
    if IMM >= 64 {
        let v_imm = vdupq_n_s64(IMM as i64 - 64i64);
        s128x2_t(vdupq_n_s64(0), vshlq_s64(a.1, v_imm))
    } else {
        let v_imm = vdupq_n_s64(IMM as i64);
        let lo = vshlq_s64(a.0, v_imm);
        let mut hi = vshlq_s64(a.1, v_imm);
        let overflow = vshlq_s64(a.0, vdupq_n_s64(-(64i64 - IMM as i64)));
        hi = vorrq_s64(hi, overflow);
        s128x2_t(lo, hi)
    }
}

#[inline]
/// Absolute value for i128
pub unsafe fn vabsq_s128(a: s128x2_t) -> s128x2_t {
    let is_neg = vcltzq_s64(a.1);
    s128x2_t(
        vbslq_s64(is_neg, vnegq_s64(a.0), a.0),
        vbslq_s64(is_neg, vnegq_s64(a.1), a.1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        unsafe {
            let value1 = vdupq_n_u64(27);
            let value2 = vdupq_n_u64(2);

            let comparison = vaddwq_u64(vmullq_u64(value1, value2), vdupq_n_u64(2));
            let flag_1 = vextractq_lo_u128::<0>(comparison);
            let control = 27 * 2 + 2;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_u64(u64::MAX);
            let value2 = vdupq_n_u64(2);

            let comparison = vaddwq_u64(vmullq_u64(value1, value2), vdupq_n_u64(2));
            let flag_1 = vextractq_q_u128::<0>(comparison);
            let control = (u64::MAX as u128 * 2) + 2;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value1 = vdupq_n_s64(27);
            let value2 = vdupq_n_s64(2);

            let comparison = vaddwq_s64(vmullq_s64(value1, value2), vdupq_n_s64(2));
            let flag_1 = vextractq_q_s128::<0>(comparison);
            let control = 27i128 * 2 + 2i128;
            assert_eq!(flag_1, control);
        }

        unsafe {
            let value1 = vdupq_n_s64(i64::MAX);
            let value2 = vdupq_n_s64(2);

            let comparison = vaddwq_s64(vmullq_s64(value1, value2), vdupq_n_s64(2));
            let flag_1 = vextractq_q_s128::<0>(comparison);
            let control = (i64::MAX as i128 * 2) + 2;
            assert_eq!(flag_1, control);
        }
    }

    #[test]
    fn test_shift_right() {
        unsafe {
            let value1 = vdupq_n_u64(27);
            let value2 = vdupq_n_u64(2);

            let comparison = vshrq_n_u128::<2>(vmullq_u64(value1, value2));
            let flag_1 = vextractq_lo_u128::<0>(comparison);
            let control = (27 * 2) >> 2;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_u64(u64::MAX);
            let value2 = vdupq_n_u64(2);

            let comparison = vshrq_n_u128::<4>(vmullq_u64(value1, value2));
            let flag_1 = vextractq_q_u128::<0>(comparison);
            let control = (u64::MAX as u128 * 2) >> 4;
            assert_eq!(flag_1, control);
        }
    }

    #[test]
    fn test_widen_mul() {
        unsafe {
            let value1 = vdupq_n_u64(27);
            let value2 = vdupq_n_u64(2);

            let comparison = vmullq_u64(value1, value2);
            let flag_1 = vextractq_q_u128::<0>(comparison);
            let control = 27 * 2;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_s64(27);
            let value2 = vdupq_n_s64(2);

            let comparison = vmullq_s64(value1, value2);
            let flag_1 = vextractq_lo_s128::<0>(comparison);
            let control = 27 * 2;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_s64(27);
            let value2 = vdupq_n_s64(-2);

            let comparison = vmullq_s64(value1, value2);
            let flag_1 = vextractq_lo_s128::<0>(comparison);
            let control = -27 * 2;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_u64(u64::MAX);
            let value2 = vdupq_n_u64(2);
            let comparison = vmullq_u64(value1, value2);
            let flag_1 = vextractq_q_u128::<0>(comparison);
            let control: u128 = u64::MAX as u128 * 2;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_s64(i64::MAX);
            let value2 = vdupq_n_s64(-2);
            let comparison = vmullq_s64(value1, value2);
            let flag_1 = vextractq_q_s128::<0>(comparison);
            let control: i128 = i64::MAX as i128 * -2i128;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_s64(15);
            let value2 = vdupq_n_s64(-2);
            let comparison = vmullq_s64(value1, value2);
            let flag_1 = vextractq_q_s128::<0>(comparison);
            let control: i128 = -30;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_s64(-15);
            let value2 = vdupq_n_s64(-2);
            let comparison = vmullq_s64(value1, value2);
            let flag_1 = vextractq_q_s128::<0>(comparison);
            let control: i128 = 30;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_s64(15);
            let value2 = vdupq_n_s64(2);
            let comparison = vmullq_s64(value1, value2);
            let flag_1 = vextractq_q_s128::<0>(comparison);
            let control: i128 = 30;
            assert_eq!(flag_1, control);
        }
    }
}
