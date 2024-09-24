/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::vhypot4q_f64;
use crate::{eabs, efmax};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

#[inline]
fn do_hypot4(x: f64, y: f64, z: f64, w: f64) -> f64 {
    let x = eabs(x);
    let y = eabs(y);
    let z = eabs(z);
    let w = eabs(w);

    let max = efmax(efmax(efmax(x, y), z), w);

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
        f64::INFINITY
    } else if x.is_nan() || y.is_nan() || z.is_nan() || w.is_nan() || ret.is_nan() {
        f64::NAN
    } else {
        ret
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn do_hypot4_neon(x: f64, y: f64, z: f64, w: f64) -> f64 {
    unsafe {
        let vx = vdupq_n_f64(x);
        let vy = vdupq_n_f64(y);
        let vz = vdupq_n_f64(z);
        let vw = vdupq_n_f64(w);
        vgetq_lane_f64::<0>(vhypot4q_f64(vx, vy, vz, vw))
    }
}

/// Computes 4D Euclidian Distance *ULP 0.6666*
#[inline]
pub fn ehypot4(x: f64, y: f64, z: f64, w: f64) -> f64 {
    let mut _dispatcher: fn(f64, f64, f64, f64) -> f64 = do_hypot4;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = do_hypot4_neon;
    }
    _dispatcher(x, y, z, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_hypot4() {
        {
            let x = 2.0f64;
            let y = 32f64;
            let z = 12f64;
            let w = 6f64;
            let ag = do_hypot4(x, y, z, w);
            assert_eq!(ag, 34.75629439396553f64);
        }
    }
}
