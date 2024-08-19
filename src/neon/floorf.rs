/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::general::{vcopysignq_f32, visinfq_f32};
use std::arch::aarch64::*;

/// Shorter and significantly faster reach skipping Inf checks
#[inline]
pub unsafe fn vfloorq_f32(x: float32x4_t) -> float32x4_t {
    let ones = vdupq_n_f32(1f32);
    let z = vcvtq_s32_f32(x);
    let r = vcvtq_f32_s32(z);

    return vbslq_f32(vcgtq_f32(r, x), vsubq_f32(r, ones), r);
}

/// Floor method complaints with f32 specification with infinity checks
#[inline]
pub unsafe fn vfloorq_ie_f32(x: float32x4_t) -> float32x4_t {
    let mut fr = vsubq_f32(x, vcvtq_f32_s32(vcvtq_s32_f32(x)));
    fr = vbslq_f32(vcltzq_f32(fr), vaddq_f32(fr, vdupq_n_f32(1f32)), fr);
    let c = vcopysignq_f32(vsubq_f32(x, fr), x);
    vbslq_f32(
        vorrq_u32(
            visinfq_f32(x),
            vcgeq_f32(vabsq_f32(x), vdupq_n_f32((1i32 << 23i32) as f32)),
        ),
        x,
        c,
    )
}
