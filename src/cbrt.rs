/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::{vdupq_n_f64, vgetq_lane_f64};

use crate::cbrtf::halley_cbrt;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::vcbrtq_f64;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::{_mm_cbrt_pd, _mm_extract_pd};
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

const B1: u32 = 715094163;

#[inline]
fn do_ecbrt(x: f64) -> f64 {
    if x == 0f64 {
        return x;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let mut ui: u64 = x.to_bits();
    let t: f64;
    let mut hx: u32 = (ui >> 32) as u32 & 0x7fffffff;

    hx = hx / 3 + B1;
    ui &= 1 << 63;
    ui |= (hx as u64) << 32;
    t = f64::from_bits(ui);
    let c1 = halley_cbrt(t, x);
    let c2 = halley_cbrt(c1, x);
    let c3 = halley_cbrt(c2, x);
    c3
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn do_cbrt_neon(d: f64) -> f64 {
    unsafe {
        let ld = vdupq_n_f64(d);
        vgetq_lane_f64::<0>(vcbrtq_f64(ld))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
fn do_cbrt_sse(d: f64) -> f64 {
    unsafe {
        let ld = _mm_set1_pd(d);
        _mm_extract_pd::<0>(_mm_cbrt_pd(ld))
    }
}

/// Computes Cube Root *ULP 2.0*
pub fn ecbrt(x: f64) -> f64 {
    let mut _dispatcher: fn(f64) -> f64 = do_ecbrt;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = do_cbrt_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_cbrt_sse;
    }
    _dispatcher(x)
}
