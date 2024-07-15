/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::general::{visinfq_f32, visnanq_f32, vmlafq_f32};
use std::arch::aarch64::*;

#[inline]
/// Method that computes 3D Euclidian distance *ULP 0.6666*
pub unsafe fn vhypot3q_f32(x: float32x4_t, y: float32x4_t, z: float32x4_t) -> float32x4_t {
    let x = vabsq_f32(x);
    let y = vabsq_f32(y);
    let z = vabsq_f32(z);
    let max = vmaxq_f32(vmaxq_f32(x, y), z);
    let norm_x = vdivq_f32(x, max);
    let norm_y = vdivq_f32(y, max);
    let norm_z = vdivq_f32(z, max);

    let accumulator = vmlafq_f32(
        norm_x,
        norm_x,
        vmlafq_f32(norm_y, norm_y, vmulq_f32(norm_z, norm_z)),
    );

    let mut ret = vmulq_f32(vsqrtq_f32(accumulator), max);
    let is_any_infinite = vorrq_u32(vorrq_u32(visinfq_f32(x), visinfq_f32(y)), visinfq_f32(z));
    let is_any_nan = vorrq_u32(visnanq_f32(x), visnanq_f32(y));
    let is_max_zero = vceqzq_f32(max);
    let is_result_nan = visnanq_f32(ret);
    ret = vbslq_f32(is_any_infinite, vdupq_n_f32(f32::INFINITY), ret);
    ret = vbslq_f32(is_any_nan, vdupq_n_f32(f32::NAN), ret);
    ret = vbslq_f32(is_max_zero, vdupq_n_f32(0f32), ret);
    ret = vbslq_f32(is_result_nan, vdupq_n_f32(f32::INFINITY), ret);
    ret
}

/// Method that computes 3D Euclidian distance *ULP 0.6666*, skipping Inf, Nan checks
#[inline]
pub unsafe fn vhypot3q_fast_f32(x: float32x4_t, y: float32x4_t, z: float32x4_t) -> float32x4_t {
    let x = vabsq_f32(x);
    let y = vabsq_f32(y);
    let z = vabsq_f32(z);
    let max = vmaxq_f32(vmaxq_f32(x, y), z);
    let norm_x = vdivq_f32(x, max);
    let norm_y = vdivq_f32(y, max);
    let norm_z = vdivq_f32(z, max);

    let accumulator = vmlafq_f32(
        norm_x,
        norm_x,
        vmlafq_f32(norm_y, norm_y, vmulq_f32(norm_z, norm_z)),
    );

    let ret = vmulq_f32(vsqrtq_f32(accumulator), max);
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypot3f() {
        unsafe {
            // Test regular
            let vx = vdupq_n_f32(3.);
            let vy = vdupq_n_f32(4.);
            let vz = vdupq_n_f32(5.);
            let comparison = vhypot3q_f32(vx, vy, vz);
            let flag_1 = vgetq_lane_f32::<0>(comparison);
            assert_eq!(flag_1, vgetq_lane_f32::<1>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<2>(comparison));
            assert_eq!(flag_1, vgetq_lane_f32::<3>(comparison));
            assert_eq!(flag_1, 7.0710678118654752440f32);
        }
    }
}
