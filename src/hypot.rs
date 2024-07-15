/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::abs::eabs;
use crate::fmax::efmax;
use crate::fmin::efmin;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vhypotq_f64;
use crate::sqrt::esqrt;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::{_mm_extract_pd, _mm_hypot_pd};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::*;
#[cfg(all(target_feature = "sse4.1", target_arch = "x86"))]
use std::arch::x86::*;
#[cfg(all(target_feature = "sse4.1", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[inline(always)]
fn do_hypot(x: f64, y: f64) -> f64 {
    let x = eabs(x);
    let y = eabs(y);
    let max = efmax(x, y);
    let min = efmin(x, y);
    let r = min / max;
    let ret = max * esqrt(1f64 + r * r);

    if (x == f64::INFINITY) || (y == f64::INFINITY) {
        f64::INFINITY
    } else if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if min == 0. {
        max
    } else if ret.is_nan() {
        f64::INFINITY
    } else {
        ret
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline]
fn do_hypot_neon(x: f64, y: f64) -> f64 {
    unsafe {
        let vx = vdupq_n_f64(x);
        let vy = vdupq_n_f64(y);
        vgetq_lane_f64::<0>(vhypotq_f64(vx, vy))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_hypot_sse(x: f64, y: f64) -> f64 {
    unsafe {
        let vx = _mm_set1_pd(x);
        let vy = _mm_set1_pd(y);
        _mm_extract_pd::<0>(_mm_hypot_pd(vx, vy))
    }
}

/// Computes 2D Euclidian Distance *ULP 0.5*
pub fn ehypot(x: f64, y: f64) -> f64 {
    let mut _dispatcher: fn(f64, f64) -> f64 = do_hypot;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_hypot_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_hypot_sse;
    }
    _dispatcher(x, y)
}
