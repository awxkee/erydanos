/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

use crate::neon::general::{visinfq_f64, visneginfq_f64, vmlafq_f64};
use crate::neon::poly128::{vaddwq_u64, vmovnq_u128, vmullq_u64, vshrq_n_u128};

#[inline(always)]
unsafe fn halley_cbrt(x: float64x2_t, a: float64x2_t) -> float64x2_t {
    let tx = vmulq_f64(vmulq_f64(x, x), x);
    let twos = vdupq_n_f64(2f64);
    let num = vmlafq_f64(twos, a, tx);
    let den = vmlafq_f64(twos, tx, a);
    let scale = vdivq_f64(num, den);
    vmulq_f64(x, scale)
}

#[inline(always)]
unsafe fn integer_pow_1_3(hx: uint64x2_t) -> uint64x2_t {
    let scale = vdupq_n_u64(341);
    let wide = vmullq_u64(hx, scale);
    let shifted = vshrq_n_u128::<10>(wide);
    let addiction = vdupq_n_u64(715094163);
    let product = vaddwq_u64(shifted, addiction);
    vmovnq_u128(product)
}

#[inline(always)]
pub unsafe fn vcbrtq_f64(x: float64x2_t) -> float64x2_t {
    let mut v = vcbrtq_fast_f64(x);
    v = vbslq_f64(visinfq_f64(x), vdupq_n_f64(f64::INFINITY), v);
    v = vbslq_f64(visneginfq_f64(x), vdupq_n_f64(f64::NEG_INFINITY), v);
    v
}

#[inline(always)]
pub unsafe fn vcbrtq_fast_f64(x: float64x2_t) -> float64x2_t {
    let mut ui = vreinterpretq_u64_f64(x);
    let hx = vandq_u64(vshrq_n_u64::<32>(ui), vdupq_n_u64(0x7fffffff));

    let hx = integer_pow_1_3(hx);

    ui = vandq_u64(ui, vdupq_n_u64(1 << 63));
    ui = vorrq_u64(ui, vshlq_n_u64::<32>(hx));

    let t = vreinterpretq_f64_u64(ui);

    let c0 = halley_cbrt(t, x);
    let c1 = halley_cbrt(c0, x);
    let c2 = halley_cbrt(c1, x);

    let v = vbslq_f64(vceqzq_f64(x), vdupq_n_f64(0f64), c2);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cbrtd() {
        unsafe {
            let value = vdupq_n_f64(27f64);
            let comparison = vcbrtq_f64(value);
            let flag_1 = vgetq_lane_f64::<1>(comparison);
            let control = 3f64;
            assert_eq!(flag_1, control);
            assert_eq!(vgetq_lane_f64::<0>(comparison), flag_1);
        }
        unsafe {
            let value = vdupq_n_f64(27f64);
            let comparison = vcbrtq_fast_f64(value);
            let flag_1 = vgetq_lane_f64::<1>(comparison);
            let control = 3f64;
            assert_eq!(flag_1, control);
            assert_eq!(vgetq_lane_f64::<0>(comparison), flag_1);
        }
        unsafe {
            let value = vdupq_n_f64(1500000000f64);
            let comparison = vcbrtq_f64(value);
            let flag_1 = vgetq_lane_f64::<1>(comparison);
            let control = 1144.7142425533316f64;
            assert_eq!(flag_1, control);
            assert_eq!(vgetq_lane_f64::<0>(comparison), flag_1);
        }
    }
}
