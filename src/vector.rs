/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};

use num_traits::AsPrimitive;

pub trait VectorFloat:
    Copy
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Add<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + AddAssign
    + Upper
    + 'static
{
}

impl VectorFloat for f32 {}

impl VectorFloat for f64 {}

#[derive(Copy, Clone)]
pub struct Vector<T: VectorFloat> {
    pub x: T,
    pub y: T,
}

impl<T: VectorFloat> Vector<T> {
    #[inline]
    pub const fn new(x: T, y: T) -> Vector<T> {
        Vector { x, y }
    }

    #[inline]
    pub fn normalize(self) -> Vector<T> {
        let jx = self.x + self.y;
        let jy = self.x - jx + self.y;
        return Vector::new(jx, jy);
    }

    #[inline]
    pub fn dfsqu(self) -> Vector<T> {
        let xh = self.x.upper();
        let xl = self.x - xh;

        let rx = self.x * self.x;
        let ry = xh * xh - rx + (xh + xh) * xl + xl * xl + self.x * (self.y + self.y);

        return Vector::new(rx, ry);
    }
}

impl<T: VectorFloat> Add<T> for Vector<T> {
    type Output = Self;
    #[inline]
    fn add(self, other: T) -> Self {
        let r0 = self.x + other;
        let v = r0 - self.x;
        Self::new(r0, self.x - (r0 - v) + (other - v) + self.y)
    }
}

impl From<f32> for Vector<f32> {
    #[inline]
    fn from(f: f32) -> Self {
        Self::new(f, 0.)
    }
}

pub trait Upper {
    fn upper(self) -> Self;
}

impl Upper for f32 {
    #[inline]
    fn upper(self) -> Self {
        return f32::from_bits(self.to_bits() & 0xfffff000);
    }
}

impl Upper for f64 {
    #[inline]
    fn upper(self) -> Self {
        return f64::from_bits(self.to_bits() & 0xfffffffff8000000);
    }
}

#[inline]
pub(crate) fn dmul_2_s<T: VectorFloat>(x: T, y: T) -> Vector<T>
where
    f32: AsPrimitive<T>,
{
    let xh = x.upper();
    let xl = x - xh;
    let yh = y.upper();
    let yl = y - yh;
    let mut r = Vector {
        x: 0f32.as_(),
        y: 0f32.as_(),
    };

    r.x = x * y;
    r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl;

    return r;
}

#[inline]
pub(crate) fn dadd_2_s_v<T: VectorFloat>(x: T, y: Vector<T>) -> Vector<T>
where
    f32: AsPrimitive<T>,
{
    let mut r = Vector {
        x: 0f32.as_(),
        y: 0f32.as_(),
    };

    r.x = x + y.x;
    let v = r.x - x;
    r.y = (x - (r.x - v)) + (y.x - v) + y.y;

    return r;
}

#[inline]
pub(crate) fn drec_2_s<T: VectorFloat>(d: T) -> Vector<T>
where
    f32: AsPrimitive<T>,
{
    let t = 1.0f32.as_() / d;
    let dh = d.upper();
    let dl = d - dh;
    let th = t.upper();
    let tl = t - th;
    let mut q = Vector {
        x: 0f32.as_(),
        y: 0f32.as_(),
    };

    q.x = t;
    q.y = t * (1f32.as_() - dh * th - dh * tl - dl * th - dl * tl);

    return q;
}

#[inline]
pub(crate) fn dmul_2<T: VectorFloat>(x: Vector<T>, y: Vector<T>) -> Vector<T>
where
    f32: AsPrimitive<T>,
{
    let xh = x.x.upper();
    let xl = x.x - xh;
    let yh = y.x.upper();
    let yl = y.x - yh;
    let mut r = Vector {
        x: 0f32.as_(),
        y: 0f32.as_(),
    };

    r.x = x.x * y.x;
    r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl + x.x * y.y + x.y * y.x;

    return r;
}

impl From<Vector<f32>> for f32 {
    #[inline]
    fn from(f: Vector<f32>) -> Self {
        f.x + f.y
    }
}
