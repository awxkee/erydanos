/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

#[inline]
pub unsafe fn vmlafq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    #[cfg(target_arch = "aarch64")]
    {
        return vfmaq_f32(c, b, a);
    }
    #[cfg(target_arch = "arm")]
    {
        return vmlaq_f32(c, b, a);
    }
}

#[inline]
pub unsafe fn vmlsfq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    vmlafq_f32(a, b, vnegq_f32(c))
}

#[inline]
pub unsafe fn vmlafq_nf_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    vaddq_f32(vmulq_f32(a, b), c)
}

#[inline]
pub unsafe fn vmlafq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    #[cfg(target_arch = "aarch64")]
    {
        return vfmaq_f64(c, b, a);
    }
    #[cfg(target_arch = "arm")]
    {
        return vmlaq_f64(c, b, a);
    }
}

#[inline]
pub unsafe fn vmlsfq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    vmlafq_f64(a, b, vnegq_f64(c))
}

#[inline]
pub unsafe fn vpow2ifq_s32(q: int32x4_t) -> int32x4_t {
    let j = vshlq_n_s32::<23>(vaddq_s32(q, vdupq_n_s32(0x7f)));
    j
}

#[inline]
pub unsafe fn vpow2ifq_s64(q: int64x2_t) -> int64x2_t {
    let j = vshlq_n_s64::<52>(vaddq_s64(q, vdupq_n_s64(0x3ff)));
    j
}

#[inline]
/// Returns true flag if value is Infinity
pub unsafe fn visinfq_f32(d: float32x4_t) -> uint32x4_t {
    return vceqq_f32(d, vdupq_n_f32(f32::INFINITY));
}

#[inline]
/// Returns true flag if value is Neg Infinity
pub unsafe fn visneginfq_f32(d: float32x4_t) -> uint32x4_t {
    return vceqq_f32(d, vdupq_n_f32(f32::NEG_INFINITY));
}

#[inline]
/// Returns true flag if value is Infinity
pub unsafe fn visinfq_f64(d: float64x2_t) -> uint64x2_t {
    return vceqq_f64(d, vdupq_n_f64(f64::INFINITY));
}

#[inline]
/// Returns true flag if value is Neg Infinity
pub unsafe fn visneginfq_f64(d: float64x2_t) -> uint64x2_t {
    return vceqq_f64(d, vdupq_n_f64(f64::NEG_INFINITY));
}

#[inline]
/// Returns true flag if value is NaN
pub unsafe fn visnanq_f64(d: float64x2_t) -> uint64x2_t {
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(d, d))));
}

#[inline]
/// Returns true flag if value is NaN
pub unsafe fn visnanq_f32(d: float32x4_t) -> uint32x4_t {
    return vmvnq_u32(vceqq_f32(d, d));
}

#[inline]
pub unsafe fn vsignbitq_f32(f: float32x4_t) -> uint32x4_t {
    return vandq_u32(
        vreinterpretq_u32_f32(f),
        vreinterpretq_u32_f32(vdupq_n_f32(-0.0f32)),
    );
}

#[inline]
pub unsafe fn vsignbitq_f64(f: float64x2_t) -> uint64x2_t {
    return vandq_u64(
        vreinterpretq_u64_f64(f),
        vreinterpretq_u64_f64(vdupq_n_f64(-0.0f64)),
    );
}

#[inline]
pub unsafe fn vmulsignq_f32(va: float32x4_t, vb: float32x4_t) -> float32x4_t {
    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(va), vsignbitq_f32(vb)))
}

#[inline]
pub unsafe fn vmulsignq_f64(va: float64x2_t, vb: float64x2_t) -> float64x2_t {
    vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(va), vsignbitq_f64(vb)))
}

#[inline]
pub unsafe fn visnegzeroq_f32(d: float32x4_t) -> uint32x4_t {
    return vceqq_f32(d, vdupq_n_f32(-0.0f32));
}

#[inline]
pub unsafe fn visnegzeroq_f64(d: float64x2_t) -> uint64x2_t {
    return vceqq_f64(d, vdupq_n_f64(-0.0f64));
}

#[inline]
/// Founds n in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn vilogb2kq_f32(d: float32x4_t) -> int32x4_t {
    vsubq_s32(
        vandq_s32(
            vshrq_n_s32::<23>(vreinterpretq_s32_f32(d)),
            vdupq_n_s32(0xff),
        ),
        vdupq_n_s32(0x7f),
    )
}

#[inline]
/// Founds a in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn vldexp3kq_f32(x: float32x4_t, n: int32x4_t) -> float32x4_t {
    vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32::<23>(n)))
}

#[inline]
/// Founds n in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn vilogb2kq_f64(d: float64x2_t) -> int64x2_t {
    vsubq_s64(
        vandq_s64(
            vshrq_n_s64::<52>(vreinterpretq_s64_f64(d)),
            vdupq_n_s64(0x7ff),
        ),
        vdupq_n_s64(0x3ff),
    )
}

#[inline]
/// Founds a in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn vldexp3kq_f64(x: float64x2_t, n: int64x2_t) -> float64x2_t {
    vreinterpretq_f64_s64(vaddq_s64(vreinterpretq_s64_f64(x), vshlq_n_s64::<52>(n)))
}

#[inline]
pub unsafe fn vcopysignq_f32(x: float32x4_t, y: float32x4_t) -> float32x4_t {
    let mag = vdupq_n_u32(1 << 31);
    let j = vbicq_u32(vreinterpretq_u32_f32(x), mag);
    let v = vandq_u32(vreinterpretq_u32_f32(y), mag);
    vreinterpretq_f32_u32(veorq_u32(j, v))
}

#[inline]
pub unsafe fn vcopysignq_f64(x: float64x2_t, y: float64x2_t) -> float64x2_t {
    let mag = vdupq_n_u64(1 << 63);
    let j = vbicq_u64(vreinterpretq_u64_f64(x), mag);
    let v = vandq_u64(vreinterpretq_u64_f64(y), mag);
    vreinterpretq_f64_u64(veorq_u64(j, v))
}

#[inline]
pub unsafe fn vmul_u64(ab: uint64x1_t, cd: uint64x1_t) -> uint64x1_t {
    /* ac = (ab & 0xFFFFFFFF) * (cd & 0xFFFFFFFF); */
    let ac = vmull_u32(vreinterpret_u32_u64(ab), vreinterpret_u32_u64(cd));

    /* b = ab >> 32; */
    let b = vshr_n_u64::<32>(ab);

    /* bc = b * (cd & 0xFFFFFFFF); */
    let bc = vmull_u32(vreinterpret_u32_u64(b), vreinterpret_u32_u64(cd));

    /* d = cd >> 32; */
    let d = vshr_n_u64::<32>(cd);

    /* ad = (ab & 0xFFFFFFFF) * d; */
    let ad = vmull_u32(vreinterpret_u32_u64(ab), vreinterpret_u32_u64(d));

    /* high = bc + ad; */
    let mut high = vadd_u64(vget_low_u64(bc), vget_low_u64(ad));

    /* high <<= 32; */
    high = vshl_n_u64::<32>(high);

    /* return ac + high; */
    return vadd_u64(high, vget_low_u64(ac));
}

#[inline]
/// Multiplies u64 together and takes low part, do not care about overflowing
pub unsafe fn vmulq_u64(ab: uint64x2_t, cd: uint64x2_t) -> uint64x2_t {
    /* ac = (ab & 0xFFFFFFFF) * (cd & 0xFFFFFFFF); */
    let ab_low = vmovn_u64(ab);
    let cd_low = vmovn_u64(cd);
    let ac = vmull_u32(ab_low, cd_low);

    /* b = ab >> 32; */
    let b = vshrq_n_u64::<32>(ab);

    /* bc = b * (cd & 0xFFFFFFFF); */
    let bc = vmull_u32(vmovn_u64(b), vmovn_u64(cd));

    /* d = cd >> 32; */
    let d = vshrq_n_u64::<32>(cd);

    /* ad = (ab & 0xFFFFFFFF) * d; */
    let ad = vmull_u32(vmovn_u64(ab), vmovn_u64(d));

    /* high = bc + ad; */
    let mut high = vaddq_u64(bc, ad);

    /* high <<= 32; */
    high = vshlq_n_u64::<32>(high);

    /* return ac + high; */
    return vaddq_u64(high, ac);
}

#[inline]
pub unsafe fn vmul_s64(ab: int64x1_t, cd: int64x1_t) -> int64x1_t {
    vreinterpret_s64_u64(vmul_u64(vreinterpret_u64_s64(ab), vreinterpret_u64_s64(cd)))
}

#[inline]
pub unsafe fn vqshrn_n_u128<const SHIFT: i32>(a: uint64x2x2_t) -> uint64x2_t {
    let high_products = vshrq_n_u64::<SHIFT>(a.1);
    let low_products = vshrq_n_u64::<SHIFT>(a.0);
    vorrq_u64(high_products, low_products)
}

#[inline]

pub unsafe fn vmulq_s64(ab: int64x2_t, cd: int64x2_t) -> int64x2_t {
    vreinterpretq_s64_u64(vmulq_u64(
        vreinterpretq_u64_s64(ab),
        vreinterpretq_u64_s64(cd),
    ))
}

#[inline]
/// Returns true flag if value is not an integral value
pub unsafe fn visnotintegralq_f32(d: float32x4_t) -> uint32x4_t {
    return vmvnq_u32(vceqq_f32(d, vcvtq_f32_s32(vcvtq_s32_f32(d))));
}

#[inline]
/// Returns true flag if value is not an integral value
pub unsafe fn visnotintegralq_f64(d: float64x2_t) -> uint64x2_t {
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(
        d,
        vcvtq_f64_s64(vcvtq_s64_f64(d)),
    ))));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_lo_64int() {
        unsafe {
            let value1 = vdupq_n_u64(27);
            let value2 = vdupq_n_u64(2);

            let comparison = vmulq_u64(value1, value2);
            let flag_1 = vgetq_lane_u64::<0>(comparison);
            let control = 27 * 2;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_s64(27);
            let value2 = vdupq_n_s64(2);

            let comparison = vmulq_s64(value1, value2);
            let flag_1 = vgetq_lane_s64::<0>(comparison);
            let control = 27 * 2;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_s64(27);
            let value2 = vdupq_n_s64(-2);

            let comparison = vmulq_s64(value1, value2);
            let flag_1 = vgetq_lane_s64::<0>(comparison);
            let control = -27 * 2;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_s64(-27);
            let value2 = vdupq_n_s64(-2);

            let comparison = vmulq_s64(value1, value2);
            let flag_1 = vgetq_lane_s64::<0>(comparison);
            let control = 27 * 2;
            assert_eq!(flag_1, control);
        }
        unsafe {
            let value1 = vdupq_n_s64(-27);
            let value2 = vdupq_n_s64(2);

            let comparison = vmulq_s64(value1, value2);
            let flag_1 = vgetq_lane_s64::<0>(comparison);
            let control = -27 * 2;
            assert_eq!(flag_1, control);
        }
    }
}
