/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[inline]
pub fn eabsf(x: f32) -> f32 {
    f32::from_bits(0x_7fff_ffff & x.to_bits())
}

#[inline]
pub fn eabs(x: f64) -> f64 {
    f64::from_bits(0x_7fff_ffff_ffff_ffff & x.to_bits())
}
