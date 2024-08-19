/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::neon::general::{visinfq_f32, visnanq_f32, vmlafq_f32};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
/// Method that computes 4D Euclidian distance *ULP 0.6666*
pub unsafe fn vhypot4q_f32(
    x: float32x4_t,
    y: float32x4_t,
    z: float32x4_t,
    w: float32x4_t,
) -> float32x4_t {
    let x = vabsq_f32(x);
    let y = vabsq_f32(y);
    let z = vabsq_f32(z);
    let w = vabsq_f32(w);
    let max = vmaxq_f32(vmaxq_f32(vmaxq_f32(x, y), z), w);
    let recip_max = vdivq_f32(vdupq_n_f32(1.), max);
    let norm_x = vmulq_f32(x, recip_max);
    let norm_y = vmulq_f32(y, recip_max);
    let norm_z = vmulq_f32(z, recip_max);
    let norm_w = vmulq_f32(w, recip_max);

    let accumulator = vmlafq_f32(
        norm_x,
        norm_x,
        vmlafq_f32(
            norm_y,
            norm_y,
            vmlafq_f32(norm_z, norm_z, vmulq_f32(norm_w, norm_w)),
        ),
    );

    let mut ret = vmulq_f32(vsqrtq_f32(accumulator), max);
    let is_any_infinite = vorrq_u32(
        vorrq_u32(vorrq_u32(visinfq_f32(x), visinfq_f32(y)), visinfq_f32(z)),
        visinfq_f32(w),
    );
    let mut is_any_nan = vorrq_u32(
        vorrq_u32(vorrq_u32(visnanq_f32(x), visnanq_f32(y)), visnanq_f32(z)),
        visnanq_f32(w),
    );
    let is_max_zero = vceqzq_f32(max);
    is_any_nan = vorrq_u32(visnanq_f32(ret), is_any_nan);
    ret = vbslq_f32(is_any_nan, vdupq_n_f32(f32::NAN), ret);
    ret = vbslq_f32(is_any_infinite, vdupq_n_f32(f32::INFINITY), ret);
    ret = vbslq_f32(is_max_zero, vdupq_n_f32(0f32), ret);
    ret
}

/// Method that computes 4D Euclidian distance *ULP 0.6666*, skipping Inf, Nan checks
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn vhypot4q_fast_f32(
    x: float32x4_t,
    y: float32x4_t,
    z: float32x4_t,
    w: float32x4_t,
) -> float32x4_t {
    let x = vabsq_f32(x);
    let y = vabsq_f32(y);
    let z = vabsq_f32(z);
    let w = vabsq_f32(w);
    let max = vmaxq_f32(vmaxq_f32(vmaxq_f32(x, y), z), w);
    let recip_max = vdivq_f32(vdupq_n_f32(1.), max);
    let norm_x = vmulq_f32(x, recip_max);
    let norm_y = vmulq_f32(y, recip_max);
    let norm_z = vmulq_f32(z, recip_max);
    let norm_w = vmulq_f32(w, recip_max);

    let accumulator = vmlafq_f32(
        norm_x,
        norm_x,
        vmlafq_f32(
            norm_y,
            norm_y,
            vmlafq_f32(norm_z, norm_z, vmulq_f32(norm_w, norm_w)),
        ),
    );

    let ret = vmulq_f32(vsqrtq_f32(accumulator), max);
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypot4f() {
        unsafe {
            // Test regular
            let vx = vdupq_n_f32(3.);
            let vy = vdupq_n_f32(4.);
            let vz = vdupq_n_f32(5.);
            let vw = vdupq_n_f32(6.);
            let comparison = vhypot4q_f32(vx, vy, vz, vw);
            let flag_1 = vgetq_lane_f32::<0>(comparison);
            assert_eq!(flag_1, vgetq_lane_f32::<1>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<2>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<3>(comparison));
            assert_eq!(flag_1, 9.27361849549570375f32);
        }

        unsafe {
            // Test regular
            let vx = vdupq_n_f32(3.);
            let vy = vdupq_n_f32(4.);
            let vz = vdupq_n_f32(5.);
            let vw = vdupq_n_f32(f32::INFINITY);
            let comparison = vhypot4q_f32(vx, vy, vz, vw);
            let flag_1 = vgetq_lane_f32::<0>(comparison);
            assert_eq!(flag_1, vgetq_lane_f32::<1>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<2>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<3>(comparison));
            assert!(flag_1.is_infinite());
        }

        unsafe {
            // Test regular
            let vx = vdupq_n_f32(3.);
            let vy = vdupq_n_f32(4.);
            let vz = vdupq_n_f32(f32::INFINITY);
            let vw = vdupq_n_f32(6.);
            let comparison = vhypot4q_f32(vx, vy, vz, vw);
            let flag_1 = vgetq_lane_f32::<0>(comparison);
            assert_eq!(flag_1, vgetq_lane_f32::<1>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<2>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<3>(comparison));
            assert!(flag_1.is_infinite());
        }

        unsafe {
            // Test regular
            let vx = vdupq_n_f32(3.);
            let vy = vdupq_n_f32(f32::INFINITY);
            let vz = vdupq_n_f32(5.);
            let vw = vdupq_n_f32(6.);
            let comparison = vhypot4q_f32(vx, vy, vz, vw);
            let flag_1 = vgetq_lane_f32::<0>(comparison);
            assert_eq!(flag_1, vgetq_lane_f32::<1>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<2>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<3>(comparison));
            assert!(flag_1.is_infinite());
        }

        unsafe {
            // Test regular
            let vx = vdupq_n_f32(f32::INFINITY);
            let vy = vdupq_n_f32(4.);
            let vz = vdupq_n_f32(5.);
            let vw = vdupq_n_f32(6.);
            let comparison = vhypot4q_f32(vx, vy, vz, vw);
            let flag_1 = vgetq_lane_f32::<0>(comparison);
            assert_eq!(flag_1, vgetq_lane_f32::<1>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<2>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<3>(comparison));
            assert!(flag_1.is_infinite());
        }
    }
}
