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
use crate::_mm_hypot4_ps;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::vhypot4q_f32;
use crate::{eabsf, efmaxf};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
#[cfg(all(target_arch = "x86", target_feature = "sse4.1"))]
use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
use std::arch::x86_64::*;

#[inline]
fn do_hypot4f(x: f32, y: f32, z: f32, w: f32) -> f32 {
    let x = eabsf(x);
    let y = eabsf(y);
    let z = eabsf(z);
    let w = eabsf(w);

    let max = efmaxf(efmaxf(efmaxf(x, y), z), w);

    if max == 0.0 {
        return 0.0;
    }

    let recip_max = 1. / max;

    let norm_x = x * recip_max;
    let norm_y = y * recip_max;
    let norm_z = z * recip_max;
    let norm_w = w * recip_max;

    let ret = max * (norm_x * norm_x + norm_y * norm_y + norm_z * norm_z + norm_w * norm_w).sqrt();

    if x.is_infinite() || y.is_infinite() || z.is_infinite() || w.is_infinite() {
        f32::INFINITY
    } else if x.is_nan() || y.is_nan() || z.is_nan() || w.is_nan() || ret.is_nan() {
        f32::NAN
    } else {
        ret
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn do_hypot4f_neon(x: f32, y: f32, z: f32, w: f32) -> f32 {
    unsafe {
        let vx = vdupq_n_f32(x);
        let vy = vdupq_n_f32(y);
        let vz = vdupq_n_f32(z);
        let vw = vdupq_n_f32(w);
        vgetq_lane_f32::<0>(vhypot4q_f32(vx, vy, vz, vw))
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
#[inline]
fn do_hypot4f_sse(x: f32, y: f32, z: f32, w: f32) -> f32 {
    unsafe {
        let vx = _mm_set1_ps(x);
        let vy = _mm_set1_ps(y);
        let vz = _mm_set1_ps(z);
        let vw = _mm_set1_ps(w);
        f32::from_bits(_mm_extract_ps::<0>(_mm_hypot4_ps(vx, vy, vz, vw)) as u32)
    }
}

/// Computes 4D Euclidian Distance *ULP 0.6666*
#[inline]
pub fn ehypot4f(x: f32, y: f32, z: f32, w: f32) -> f32 {
    let mut _dispatcher: fn(f32, f32, f32, f32) -> f32 = do_hypot4f;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = do_hypot4f_neon;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = do_hypot4f_sse;
    }
    _dispatcher(x, y, z, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_hypot4f() {
        {
            let x = 2.0f32;
            let y = 32f32;
            let z = 12f32;
            let w = 6f32;
            let ag = do_hypot4f(x, y, z, w);
            assert_eq!(ag, 34.75629439396553f32);
        }
    }
}
