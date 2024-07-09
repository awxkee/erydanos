/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::abs::eabs;
use crate::asin::easin;

#[inline]
fn do_acos(x: f64) -> f64 {
    if x > 0f64 {
        std::f64::consts::FRAC_PI_2 - easin(x)
    } else {
        let v = eabs(x);
        std::f64::consts::FRAC_PI_2 + easin(v)
    }
}

#[inline]
pub fn eacos(d: f64) -> f64 {
    let mut _dispatcher: fn(f64) -> f64 = do_acos;
    _dispatcher(d)
}
