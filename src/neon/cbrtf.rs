/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::general::{visinfq_f32, visneginfq_f32, vmlafq_f32};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn halley_cbrt(x: float32x4_t, a: float32x4_t) -> float32x4_t {
    let tx = vmulq_f32(vmulq_f32(x, x), x);
    let twos = vdupq_n_f32(2f32);
    let num = vmlafq_f32(twos, a, tx);
    let den = vmlafq_f32(twos, tx, a);
    let scale = vdivq_f32(num, den);
    vmulq_f32(x, scale)
}

#[inline(always)]
unsafe fn integer_pow_1_3(hx: uint32x4_t) -> uint32x4_t {
    let scale = vdupq_n_u32(341);
    let hi = vshrq_n_u64::<10>(vmull_high_u32(hx, scale));
    let lo = vshrq_n_u64::<10>(vmull_u32(vget_low_u32(hx), vget_low_u32(scale)));
    vcombine_u32(vmovn_u64(lo), vmovn_u64(hi))
}

/// Takes cube root from value *ULP 1.5*
#[inline(always)]
pub unsafe fn vcbrtq_f32(x: float32x4_t) -> float32x4_t {
    let mut ui = vreinterpretq_u32_f32(x);
    let hx = vandq_u32(ui, vdupq_n_u32(0x7fffffff));

    let hx = vaddq_u32(integer_pow_1_3(hx), vdupq_n_u32(709958130));

    ui = vandq_u32(ui, vdupq_n_u32(0x80000000));
    ui = vorrq_u32(ui, hx);

    let t = vreinterpretq_f32_u32(ui);

    let c0 = halley_cbrt(t, x);
    let c1 = halley_cbrt(c0, x);
    let mut v = vbslq_f32(vceqzq_f32(x), vdupq_n_f32(0f32), c1);
    v = vbslq_f32(visinfq_f32(x), vdupq_n_f32(f32::INFINITY), v);
    v = vbslq_f32(visneginfq_f32(x), vdupq_n_f32(f32::NEG_INFINITY), v);
    v
}
