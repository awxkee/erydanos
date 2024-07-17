/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::abs::eabs;
use crate::asin::easin;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::vacosq_f64;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::{_mm_acos_pd, _mm_extract_pd};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::*;
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

#[inline]
fn do_acos(x: f64) -> f64 {
    if x > 0f64 {
        std::f64::consts::FRAC_PI_2 - easin(x)
    } else {
        let v = eabs(x);
        std::f64::consts::FRAC_PI_2 + easin(v)
    }
}

#[inline]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
fn do_acos_neon(d: f64) -> f64 {
    unsafe {
        let ld = vdupq_n_f64(d);
        vgetq_lane_f64::<0>(vacosq_f64(ld))
    }
}

#[inline]
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
fn do_acos_sse(d: f64) -> f64 {
    unsafe {
        let ld = _mm_set1_pd(d);
        _mm_extract_pd::<0>(_mm_acos_pd(ld))
    }
}

#[inline]
/// Computes acos for an argument, *ULP 2.0*
pub fn eacos(d: f64) -> f64 {
    let mut _dispatcher: fn(f64) -> f64 = do_acos;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_acos_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_acos_sse;
    }
    _dispatcher(d)
}
