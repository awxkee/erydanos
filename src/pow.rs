/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::abs::eabs;
use crate::efloor;
use crate::exp::eexp;
use crate::generalf::{copysignk, is_neg_infinite, is_pos_infinite};
use crate::ln::eln;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::vpowq_f64;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::{_mm_extract_pd, _mm_pow_pd};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::{vdupq_n_f64, vgetq_lane_f64};
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

#[inline]
fn do_pow(d: f64, n: f64) -> f64 {
    let value = eabs(d);
    let mut c = eexp(n * eln(value));
    c = copysignk(c, d);
    if d < 0. && efloor(n) != n {
        return f64::NAN;
    }
    if is_pos_infinite(n) || d.is_infinite() {
        f64::INFINITY
    } else if is_neg_infinite(n) {
        0f64
    } else if n.is_nan() || d.is_nan() {
        f64::NAN
    } else {
        c
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline]
fn do_pow_neon(d: f64, n: f64) -> f64 {
    unsafe {
        let val = vdupq_n_f64(d);
        let power = vdupq_n_f64(n);
        vgetq_lane_f64::<0>(vpowq_f64(val, power))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_pow_sse(d: f64, n: f64) -> f64 {
    unsafe {
        let val = _mm_set1_pd(d);
        let power = _mm_set1_pd(n);
        _mm_extract_pd::<0>(_mm_pow_pd(val, power))
    }
}

/// Computes power function, error bound *ULP 2.0*
#[inline]
pub fn epow(d: f64, n: f64) -> f64 {
    let mut _dispatcher: fn(f64, f64) -> f64 = do_pow;
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
