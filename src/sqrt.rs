/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::vector::{dadd_2_s_v, dmul_2, dmul_2_s, drec_2_s};

pub fn esqrt(d: f64) -> f64 {
    let mut q = 0.5;

    let mut d = if d < 0f64 { f64::NAN } else { d };

    if d < 8.636168555094445E-78 {
        d *= 1.157920892373162E77;
        q = 2.9387358770557188E-39 * 0.5;
    }

    if d > 1.3407807929942597e+154 {
        d *= 7.4583407312002070e-155;
        q = 1.1579208923731620e+77 * 0.5;
    }

    // http://en.wikipedia.org/wiki/Fast_inverse_square_root
    let mut x = f64::from_bits(0x5fe6ec85e7de30da - ((d + 1e-320).to_bits() >> 1));

    x = x * (1.5 - 0.5 * d * x * x);
    x = x * (1.5 - 0.5 * d * x * x);
    x = x * (1.5 - 0.5 * d * x * x) * d;

    let d2 = dmul_2(dadd_2_s_v(d, dmul_2_s(x, x)), drec_2_s(x));
    let ret = (d2.x + d2.y) * q;

    let mut ret = if d == f64::INFINITY {
        f64::INFINITY
    } else {
        ret
    };
    ret = if d == 0f64 { d } else { ret };

    return ret;
}
