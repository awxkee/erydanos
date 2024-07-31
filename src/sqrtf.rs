/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::vector::{dadd_2_s_v, dmul_2, dmul_2_s, drec_2_s};

/// Computes square root, error bound *ULP 1.0*
pub fn esqrtf(d: f32) -> f32 {
    let mut q = 0.5f32;

    let mut d = if d < 0f32 { f32::NAN } else { d };

    if d < 5.2939559203393770e-23f32 {
        d *= 1.8889465931478580e+22f32;
        q = 7.2759576141834260e-12f32 * 0.5f32;
    }

    if d > 1.8446744073709552e+19f32 {
        d *= 5.4210108624275220e-20f32;
        q = 4294967296.0f32 * 0.5f32;
    }

    // http://en.wikipedia.org/wiki/Fast_inverse_square_root
    let mut x = f32::from_bits(0x5f375a86 - ((d + 1e-45f32).to_bits() >> 1));

    x = x * (1.5f32 - 0.5f32 * d * x * x);
    x = x * (1.5f32 - 0.5f32 * d * x * x);
    x = x * (1.5f32 - 0.5f32 * d * x * x) * d;

    let d2 = dmul_2(dadd_2_s_v(d, dmul_2_s(x, x)), drec_2_s(x));

    let ret = (d2.x + d2.y) * q;

    let mut ret = if d == f32::INFINITY {
        f32::INFINITY
    } else {
        ret
    };
    ret = if d == 0f32 { d } else { ret };

    return ret;
}
