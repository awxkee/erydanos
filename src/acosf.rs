/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::*;

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::_mm_acos_ps;
use crate::abs::eabsf;
use crate::asinf::easinf;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::vacosq_f32;
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

#[inline]
fn do_acosf(x: f32) -> f32 {
    if x > 0f32 {
        std::f32::consts::FRAC_PI_2 - easinf(x)
    } else {
        let v = eabsf(x);
        std::f32::consts::FRAC_PI_2 + easinf(v)
    }
}

#[inline]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
fn do_acosf_neon(d: f32) -> f32 {
    unsafe {
        let ld = vdupq_n_f32(d);
        vgetq_lane_f32::<0>(vacosq_f32(ld))
    }
}

#[inline]
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
fn do_acosf_sse(d: f32) -> f32 {
    unsafe {
        let ld = _mm_set1_ps(d);
        f32::from_bits(_mm_extract_ps::<0>(_mm_acos_ps(ld)) as u32)
    }
}

/// Computes acos for an argument, *ULP 2.0*
#[inline]
pub fn eacosf(d: f32) -> f32 {
    let mut _dispatcher: fn(f32) -> f32 = do_acosf;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_acosf_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_acosf_sse;
    }
    _dispatcher(d)
}
