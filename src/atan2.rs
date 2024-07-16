/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

use crate::atan::eatan;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::{_mm_atan2_pd, _mm_extract_pd};

#[inline]
fn do_atan2(y: f64, x: f64) -> f64 {
    if x == 0. {
        if y > 0. {
            return std::f64::consts::FRAC_PI_2;
        }
        if y < 0. {
            return -std::f64::consts::FRAC_PI_2;
        }
        if y == 0. {
            return 0f64;
        }
    }
    let rad = eatan(y / x);
    return if x > 0f64 {
        rad
    } else if x < 0f64 && y >= 0f64 {
        std::f64::consts::PI + rad
    } else {
        // if x < 0. && y < 0.
        -std::f64::consts::PI + rad
    };
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_atan2_sse(y: f64, x: f64) -> f64 {
    unsafe {
        let vx = _mm_set1_pd(x);
        let vy = _mm_set1_pd(y);
        _mm_extract_pd::<0>(_mm_atan2_pd(vy, vx))
    }
}

#[inline]
pub fn eatan2(y: f64, x: f64) -> f64 {
    let mut _dispatcher: fn(f64, f64) -> f64 = do_atan2;
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_atan2_sse;
    }
    _dispatcher(y, x)
}
