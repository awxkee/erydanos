/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::ops::{Add, Mul, Sub};

pub trait UpperPart {
    fn upper(self) -> Self;
}

#[inline(always)]
fn upperf(o: f32) -> f32 {
    f32::from_bits(o.to_bits() & 0x_ffff_f000)
}

#[inline]
fn upper(o: f64) -> f64 {
    f64::from_bits(o.to_bits() & 0x_ffff_ffff_f800_0000)
}

impl UpperPart for f32 {
    #[inline(always)]
    fn upper(self) -> Self {
        upperf(self)
    }
}

impl UpperPart for f64 {
    #[inline(always)]
    fn upper(self) -> Self {
        upper(self)
    }
}

impl UpperPart for u64 {
    fn upper(self) -> Self {
        self & 0x_ffff_ffff_f800_0000
    }
}

impl UpperPart for i64 {
    fn upper(self) -> Self {
        (self as u64 & 0x_ffff_ffff_f800_0000) as i64
    }
}

#[inline(always)]
pub fn multiply_as_doubled<
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + UpperPart,
>(
    a: T,
    b: T,
) -> (T, T) {
    let xh = a.upper();
    let xl = a - xh;
    let yh = b.upper();
    let yl = b - yh;
    let r0 = a * b;
    let d0 = r0;
    let j0 = xh * yh;
    let j1 = xl * yh;
    let j2 = xh * yl;
    let j3 = xl * yl;
    let d1 = j0 + j1 + j2 + j3 - r0;
    (d0, d1)
}

#[inline(always)]
pub fn mul_doubled<T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + UpperPart>(
    a: (T, T),
    b: (T, T),
) -> (T, T) {
    let xh = a.0.upper();
    let xl = a.0 - xh;
    let yh = b.0.upper();
    let yl = b.0 - yh;
    let r0 = a.0 * b.0;
    (
        r0,
        xh * yh + xl * yh + xh * yl + xl * yl + a.0 * b.1 + a.1 * b.0 - r0,
    )
}
