/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use std::arch::aarch64::{
    float32x4_t, vcvtq_f32_s32, vcvtq_s32_f32, vmulq_f32, vrecpeq_f32, vsubq_f32,
};

#[inline(always)]
/// Compute fmod
pub unsafe fn vfmodq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let dividend_vec = a;
    let divisor_vec = b;
    let division = vmulq_f32(dividend_vec, vrecpeq_f32(divisor_vec));
    let int_part = vcvtq_f32_s32(vcvtq_s32_f32(division));
    let product = vmulq_f32(int_part, divisor_vec);
    let remainder = vsubq_f32(dividend_vec, product);
    remainder
}
