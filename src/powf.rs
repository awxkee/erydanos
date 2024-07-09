/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::abs::eabsf;
use crate::expf::eexpf;
use crate::generalf::{copysignfk, is_neg_infinitef, is_pos_infinitef};
use crate::lnf::elnf;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::powf::vpowq_f32;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};

#[inline]
fn do_pow(d: f32, n: f32) -> f32 {
    let value = eabsf(d);
    let mut c = eexpf(n * elnf(value));
    c = copysignfk(c, d);
    if is_pos_infinitef(n) || d.is_infinite() {
        f32::INFINITY
    } else if is_neg_infinitef(n) {
        0f32
    } else if n.is_nan() || d.is_nan() {
        f32::NAN
    } else {
        c
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline]
pub fn do_pow_neon(d: f32, n: f32) -> f32 {
    unsafe {
        let val = vdupq_n_f32(d);
        let power = vdupq_n_f32(n);
        vgetq_lane_f32::<0>(vpowq_f32(val, power))
    }
}

#[inline]
pub fn epowf(d: f32, n: f32) -> f32 {
    let mut _dispatcher: fn(f32, f32) -> f32 = do_pow;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_pow_neon;
    }
    _dispatcher(d, n)
}
