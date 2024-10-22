/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::general::{visinfq_f64, visnanq_f64, vmlafq_f64};
use std::arch::aarch64::*;

#[inline]
/// Method that computes 3D Euclidian distance *ULP 0.6666*
pub unsafe fn vhypot3q_f64(x: float64x2_t, y: float64x2_t, z: float64x2_t) -> float64x2_t {
    let x = vabsq_f64(x);
    let y = vabsq_f64(y);
    let z = vabsq_f64(z);
    let max = vmaxq_f64(vmaxq_f64(x, y), z);
    let recip_max = vdivq_f64(vdupq_n_f64(1.), max);
    let norm_x = vmulq_f64(x, recip_max);
    let norm_y = vmulq_f64(y, recip_max);
    let norm_z = vmulq_f64(z, recip_max);

    let accumulator = vmlafq_f64(
        norm_x,
        norm_x,
        vmlafq_f64(norm_y, norm_y, vmulq_f64(norm_z, norm_z)),
    );

    let mut ret = vmulq_f64(vsqrtq_f64(accumulator), max);
    let is_any_infinite = vorrq_u64(vorrq_u64(visinfq_f64(x), visinfq_f64(y)), visinfq_f64(z));
    let mut is_any_nan = vorrq_u64(vorrq_u64(visnanq_f64(x), visnanq_f64(y)), visnanq_f64(z));
    let is_max_zero = vceqzq_f64(max);
    is_any_nan = vorrq_u64(visnanq_f64(ret), is_any_nan);
    ret = vbslq_f64(is_any_infinite, vdupq_n_f64(f64::INFINITY), ret);
    ret = vbslq_f64(is_any_nan, vdupq_n_f64(f64::NAN), ret);
    ret = vbslq_f64(is_max_zero, vdupq_n_f64(0f64), ret);
    ret
}

/// Method that computes 3D Euclidian distance *ULP 0.6666*, skipping Inf, Nan checks
#[inline]
pub unsafe fn vhypot3q_fast_f64(x: float64x2_t, y: float64x2_t, z: float64x2_t) -> float64x2_t {
    let x = vabsq_f64(x);
    let y = vabsq_f64(y);
    let z = vabsq_f64(z);
    let max = vmaxq_f64(vmaxq_f64(x, y), z);
    let recip_max = vdivq_f64(vdupq_n_f64(1.), max);
    let norm_x = vmulq_f64(x, recip_max);
    let norm_y = vmulq_f64(y, recip_max);
    let norm_z = vmulq_f64(z, recip_max);

    let is_max_zero = vceqzq_f64(max);

    let accumulator = vmlafq_f64(
        norm_x,
        norm_x,
        vmlafq_f64(norm_y, norm_y, vmulq_f64(norm_z, norm_z)),
    );

    let mut ret = vmulq_f64(vsqrtq_f64(accumulator), max);
    ret = vbslq_f64(is_max_zero, vdupq_n_f64(0f64), ret);
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypot3f() {
        unsafe {
            // Test regular
            let vx = vdupq_n_f64(3.);
            let vy = vdupq_n_f64(4.);
            let vz = vdupq_n_f64(5.);
            let comparison = vhypot3q_f64(vx, vy, vz);
            let flag_1 = vgetq_lane_f64::<0>(comparison);
            assert_eq!(flag_1, vgetq_lane_f64::<1>(comparison));
            assert_eq!(flag_1, 7.0710678118654752440f64);
        }
    }
}
