/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::floor::efloorf;
use num_traits::MulAdd;
use std::ops::{Add, Mul};

/// Multiplies sign of numbers
#[inline]
pub fn mulsignf(x: f32, y: f32) -> f32 {
    f32::from_bits(x.to_bits() ^ (y.to_bits() & (1 << 31)))
}

/// Multiplies sign of numbers
#[inline]
pub fn mulsign(x: f64, y: f64) -> f64 {
    f64::from_bits(x.to_bits() ^ (y.to_bits() & (1 << 63)))
}

#[inline(always)]
pub fn make_negf(x: f32) -> f32 {
    f32::from_bits(x.to_bits() | (1 << 31))
}

#[inline(always)]
pub fn make_neg(x: f64) -> f64 {
    f64::from_bits(x.to_bits() | (1u64 << 63u64))
}

/// Copies sign from `y` to `x`
#[inline]
pub fn copysignfk(x: f32, y: f32) -> f32 {
    f32::from_bits((x.to_bits() & !(1 << 31)) ^ (y.to_bits() & (1 << 31)))
}

/// Copies sign from `y` to `x`
#[inline]
pub fn copysignk(x: f64, y: f64) -> f64 {
    f64::from_bits((x.to_bits() & !(1 << 63)) ^ (y.to_bits() & (1 << 63)))
}

/// Sign of a number
#[inline]
pub fn signf(d: f32) -> f32 {
    mulsignf(1., d)
}

/// Round towards whole integral number
#[inline]
pub fn rintfk(x: f32) -> f32 {
    (if x < 0. { x - 0.5 } else { x + 0.5 }) as i32 as f32
}

/// Round towards whole integral number
#[inline]
pub fn rintk(x: f64) -> f64 {
    (if x < 0. { x - 0.5 } else { x + 0.5 }) as i64 as f64
}

/// Computes `x*y + z` using `fma` when available
#[inline]
pub fn mlaf<T: Copy + Add<Output = T> + MulAdd + Mul<Output = T>>(x: T, y: T, z: T) -> T {
    return x * y + z;
}

/// Computes `x*y + z` using `fma` when available
#[inline]
#[cfg(target_feature = "fma")]
pub fn mlaf<T: Copy + Add<Output = T> + MulAdd + Mul<Output = T>>(x: T, y: T, z: T) -> T {
    x.mul_add(y, z)
}

#[inline]
// Founds n in ln(𝑥)=ln(𝑎)+𝑛ln(2)
pub fn ilogb2kf(d: f32) -> i32 {
    (((d.to_bits() as i32) >> 23) & 0xff) - 0x7f
}

#[inline]
pub fn ilogb2k(d: f64) -> i32 {
    (((d.to_bits() >> 52) & 0x7ff) as i32) - 0x3ff
}

#[inline]
pub fn ldexp3k(d: f64, e: i32) -> f64 {
    f64::from_bits(((d.to_bits() as i64) + ((e as i64) << 52)) as u64)
}

/// Computes 2^n
#[inline]
pub fn pow2if(q: i32) -> f32 {
    f32::from_bits(((q + 0x7f) as u32) << 23)
}

/// Checks if values is positive infinity
#[inline]
pub fn is_pos_infinite(j: f64) -> bool {
    j == f64::INFINITY
}

/// Checks if values is negative infinity
#[inline]
pub fn is_neg_infinite(j: f64) -> bool {
    j == f64::NEG_INFINITY
}

/// Checks if values is positive infinity
#[inline]
pub fn is_pos_infinitef(j: f32) -> bool {
    j == f32::INFINITY
}

/// Checks if values is negative infinity
#[inline]
pub fn is_neg_infinitef(j: f32) -> bool {
    j == f32::NEG_INFINITY
}

#[inline]
pub fn ldexp2kf(d: f32, e: i32) -> f32 {
    d * pow2if(e >> 1) * pow2if(e - (e >> 1))
}

#[inline]
// Founds a in x=a+𝑛ln(2)
pub fn ldexp3kf(d: f32, n: i32) -> f32 {
    f32::from_bits(((d.to_bits() as i32) + (n << 23)) as u32)
}

pub fn rempif2(x: f32) -> f32 {
    let n = efloorf(x / std::f32::consts::PI);
    return x - n * std::f32::consts::PI;
}

/// Checks if values is negative zero
pub trait IsNegZero {
    fn isnegzero(self) -> bool;
}

impl IsNegZero for f32 {
    #[inline]
    fn isnegzero(self) -> bool {
        self.to_bits() == (-0.0f32).to_bits()
    }
}

impl IsNegZero for f64 {
    #[inline]
    fn isnegzero(self) -> bool {
        self.to_bits() == (-0.0f64).to_bits()
    }
}

/// Computes 2^n
#[inline(always)]
pub fn pow2i(q: i32) -> f64 {
    f64::from_bits(((q + 0x3ff) as u64) << 52)
}
