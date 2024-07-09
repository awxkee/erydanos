/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::exp::vexpq_f64;
use crate::neon::general::{vcopysignq_f64, visinfq_f64, visnanq_f64, visneginfq_f64};
use crate::neon::ln::vlnq_f64;
use std::arch::aarch64::*;

#[inline]
/// Method computes pow for NEON, with Inf, Nan checks
pub unsafe fn vpowq_f64(d: float64x2_t, n: float64x2_t) -> float64x2_t {
    let value = vabsq_f64(d);
    let mut c = vexpq_f64(vmulq_f64(n, vlnq_f64(value)));
    c = vcopysignq_f64(c, d);
    let is_infinity = vorrq_u64(visinfq_f64(d), vorrq_u64(visinfq_f64(n), visneginfq_f64(n)));
    let is_power_neg_infinity = visneginfq_f64(n);
    let is_any_nan = vorrq_u64(visnanq_f64(d), visnanq_f64(n));
    let mut ret = vbslq_f64(is_infinity, vdupq_n_f64(f64::INFINITY), c);
    ret = vbslq_f64(is_power_neg_infinity, vdupq_n_f64(0f64), ret);
    ret = vbslq_f64(is_any_nan, vdupq_n_f64(f64::NAN), ret);
    ret
}
