/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::vhypot3q_f32;
use crate::{eabsf, efmaxf};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::_mm_hypot3_ps;
#[cfg(all(target_feature = "sse4.1", target_arch = "x86"))]
use std::arch::x86::*;
#[cfg(all(target_feature = "sse4.1", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[inline]
fn do_hypot3f(x: f32, y: f32, z: f32) -> f32 {
    let x = eabsf(x);
    let y = eabsf(y);
    let z = eabsf(z);

    let max = efmaxf(efmaxf(x, y), z);

    if max == 0.0 {
        return 0.0;
    }

    let recip_max = 1. / max;

    let norm_x = x * recip_max;
    let norm_y = y * recip_max;
    let norm_z = z * recip_max;

    let ret = max * (norm_x * norm_x + norm_y * norm_y + norm_z * norm_z).sqrt();

    if x == f32::INFINITY || y == f32::INFINITY || z == f32::INFINITY {
        f32::INFINITY
    } else if x.is_nan() || y.is_nan() || z.is_nan() || ret.is_nan() {
        f32::NAN
    } else {
        ret
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn do_hypot3f_neon(x: f32, y: f32, z: f32) -> f32 {
    unsafe {
        let vx = vdupq_n_f32(x);
        let vy = vdupq_n_f32(y);
        let vz = vdupq_n_f32(z);
        vgetq_lane_f32::<0>(vhypot3q_f32(vx, vy, vz))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_hypot3f_sse(x: f32, y: f32, z: f32) -> f32 {
    unsafe {
        let vx = _mm_set1_ps(x);
        let vy = _mm_set1_ps(y);
        let vz = _mm_set1_ps(z);
        f32::from_bits(_mm_extract_ps::<0>(_mm_hypot3_ps(vx, vy, vz)) as u32)
    }
}

/// Computes 3D Euclidian Distance *ULP 0.6666*
#[inline]
pub fn ehypot3f(x: f32, y: f32, z: f32) -> f32 {
    let mut _dispatcher: fn(f32, f32, f32) -> f32 = do_hypot3f;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = do_hypot3f_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_hypot3f_sse;
    }
    _dispatcher(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_hypot3f() {
        // Test regular
        let x = 2.0f32;
        let y = 32f32;
        let z = 12f32;
        let ag = do_hypot3f(x, y, z);
        assert_eq!(ag, 34.234486);
    }
}
