/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::atan::eatan;

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

#[inline]
pub fn eatan2(y: f64, x: f64) -> f64 {
    let _dispatcher: fn(f64, f64) -> f64 = do_atan2;
    _dispatcher(y, x)
}
