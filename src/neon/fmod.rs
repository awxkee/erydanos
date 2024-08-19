/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

#[inline]
/// Compute fmod for f64
pub unsafe fn vfmodq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    let dividend_vec = a;
    let divisor_vec = b;
    let division = vmulq_f64(dividend_vec, vdivq_f64(vdupq_n_f64(1.), divisor_vec));
    let int_part = vcvtq_f64_s64(vcvtq_s64_f64(division));
    let product = vmulq_f64(int_part, divisor_vec);
    let remainder = vsubq_f64(dividend_vec, product);
    remainder
}
