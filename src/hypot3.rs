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
use crate::vhypot3q_f64;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::{_mm_extract_pd, _mm_hypot3_pd};
use crate::{eabs, efmax};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::*;
#[cfg(all(target_feature = "sse4.1", target_arch = "x86"))]
use std::arch::x86::*;
#[cfg(all(target_feature = "sse4.1", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[inline]
fn do_hypot3(x: f64, y: f64, z: f64) -> f64 {
    let x = eabs(x);
    let y = eabs(y);
    let z = eabs(z);

    // Find the maximum absolute value among a, b, c
    let max = efmax(efmax(x, y), z);

    if max == 0.0 {
        return 0.0;
    }

    let norm_x = x / max;
    let norm_y = y / max;
    let norm_z = z / max;

    let ret = max * (norm_x * norm_x + norm_y * norm_y + norm_z * norm_z).sqrt();

    if x == f64::INFINITY || y == f64::INFINITY || z == f64::INFINITY {
        f64::INFINITY
    } else if x.is_nan() || y.is_nan() || z.is_nan() {
        f64::NAN
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
fn do_hypot3_neon(x: f64, y: f64, z: f64) -> f64 {
    unsafe {
        let vx = vdupq_n_f64(x);
        let vy = vdupq_n_f64(y);
        let vz = vdupq_n_f64(z);
        vgetq_lane_f64::<0>(vhypot3q_f64(vx, vy, vz))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_hypot3_sse(x: f64, y: f64, z: f64) -> f64 {
    unsafe {
        let vx = _mm_set1_pd(x);
        let vy = _mm_set1_pd(y);
        let vz = _mm_set1_pd(z);
        _mm_extract_pd::<0>(_mm_hypot3_pd(vx, vy, vz))
    }
}

/// Computes 3D Euclidian Distance *ULP 0.6666*
#[inline]
pub fn ehypot3(x: f64, y: f64, z: f64) -> f64 {
    let mut _dispatcher: fn(f64, f64, f64) -> f64 = do_hypot3;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_hypot3_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_hypot3_sse;
    }
    _dispatcher(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_hypot3() {
        // Test regular
        let x = 2.0f64;
        let y = 32f64;
        let z = 12f64;
        let ag = ehypot3(x, y, z);
        assert_eq!(ag, 34.2344855372473795263059045f64);
    }
}
