/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::_mm_pow_ps;
use crate::abs::eabsf;
use crate::efloorf;
use crate::expf::eexpf;
use crate::generalf::{copysignfk, is_neg_infinitef, is_pos_infinitef};
use crate::lnf::elnf;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vpowq_f32;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

#[inline]
fn do_pow(d: f32, n: f32) -> f32 {
    let value = eabsf(d);
    let mut c = eexpf(n * elnf(value));
    c = copysignfk(c, d);
    if d < 0. && efloorf(n) != n {
        return f32::NAN;
    }
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

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
pub fn do_pow_sse(d: f32, n: f32) -> f32 {
    unsafe {
        let val = _mm_set1_ps(d);
        let power = _mm_set1_ps(n);
        let gt = _mm_extract_ps::<0>(_mm_pow_ps(val, power)) as u32;
        f32::from_bits(gt)
    }
}

/// Computes power function, error bound *ULP 2.0*
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
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_pow_sse;
    }
    _dispatcher(d, n)
}
