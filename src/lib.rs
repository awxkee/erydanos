/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

mod abs;
pub mod acos;
pub mod acosf;
pub mod asin;
pub mod asinf;
pub mod atan;
pub mod atan2;
pub mod atan2f;
pub mod atanf;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx;
pub mod cbrt;
pub mod cbrtf;
pub mod ceil;
pub mod ceilf;
pub mod cos;
pub mod cosf;
pub mod double_precision;
pub mod exp;
pub mod expf;
pub mod floor;
mod fmax;
mod fmaxf;
mod fmin;
mod fminf;
mod generalf;
mod hypot;
mod hypot3;
mod hypot3f;
mod hypot4;
mod hypot4f;
mod hypotf;
mod ln;
mod lnf;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
mod neon;
mod pow;
mod powf;
mod shuffle;
mod sin;
mod sinf;
mod sqrt;
mod sqrtf;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod sse;
mod tan;
mod tanf;
mod vector;

pub use abs::{eabs, eabsf};
pub use acos::eacos;
pub use acosf::eacosf;
pub use asin::easin;
pub use asinf::easinf;
pub use atan::eatan;
pub use atan2::eatan2;
pub use atan2f::eatan2f;
pub use atanf::eatanf;
pub use cbrt::ecbrt;
pub use cbrtf::ecbrtf;
pub use cos::ecos;
pub use cosf::ecosf;
pub use exp::eexp;
pub use expf::eexpf;
pub use floor::{efloor, efloorf};
pub use fmax::efmax;
pub use fmaxf::efmaxf;
pub use fmin::efmin;
pub use fminf::efminf;
pub use generalf::*;
pub use hypot::ehypot;
pub use hypot3f::ehypot3f;
pub use hypot4::ehypot4;
pub use hypot4f::ehypot4f;
pub use hypotf::ehypotf;
pub use ln::eln;
pub use lnf::elnf;
pub use pow::epow;
pub use powf::epowf;
pub use sin::esin;
pub use sinf::esinf;
pub use sqrt::esqrt;
pub use sqrtf::esqrtf;
pub use tan::etan;
pub use tanf::etanf;

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
pub use neon::*;

use crate::hypot3::ehypot3;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use sse::*;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use avx::*;

pub trait Sqrtf {
    /// Computes square root
    fn esqrt(self) -> Self;
}

pub trait Cosine {
    /// Computes cosine
    fn ecos(self) -> Self;
}

impl Cosine for f32 {
    fn ecos(self) -> Self {
        ecosf(self)
    }
}

impl Cosine for f64 {
    fn ecos(self) -> Self {
        ecos(self)
    }
}

pub trait Sine {
    /// Computes sine
    fn esin(self) -> Self;
}

impl Sine for f32 {
    fn esin(self) -> Self {
        esinf(self)
    }
}

impl Sine for f64 {
    fn esin(self) -> Self {
        esin(self)
    }
}

pub trait Exponential {
    /// Computes exponent
    fn eexp(self) -> Self;
}

impl Exponential for f32 {
    fn eexp(self) -> Self {
        eexpf(self)
    }
}

impl Exponential for f64 {
    fn eexp(self) -> Self {
        eexp(self)
    }
}

impl Sqrtf for f32 {
    fn esqrt(self) -> Self {
        esqrtf(self)
    }
}

impl Sqrtf for f64 {
    fn esqrt(self) -> Self {
        esqrt(self)
    }
}

pub trait Roundable {
    /// Rounds value towards infinity
    fn efloor(self) -> Self;
}

impl Roundable for f32 {
    fn efloor(self) -> Self {
        efloorf(self)
    }
}

impl Roundable for f64 {
    fn efloor(self) -> Self {
        efloor(self)
    }
}

pub trait Signed {
    /// Modulo operation
    fn eabs(self) -> Self;
}

impl Signed for f32 {
    fn eabs(self) -> Self {
        eabsf(self)
    }
}

impl Signed for f64 {
    fn eabs(self) -> Self {
        eabs(self)
    }
}

pub trait Logarithmic {
    /// Computes natural logarithm for value
    fn eln(self) -> Self;
}

impl Logarithmic for f32 {
    fn eln(self) -> Self {
        elnf(self)
    }
}

impl Logarithmic for f64 {
    fn eln(self) -> Self {
        eln(self)
    }
}

pub trait Tangent {
    /// Computes tan for value
    fn etan(self) -> Self;
}

impl Tangent for f32 {
    fn etan(self) -> Self {
        etanf(self)
    }
}

impl Tangent for f64 {
    fn etan(self) -> Self {
        etan(self)
    }
}

pub trait Power {
    /// Computes power for given value and power
    fn epow(self, n: Self) -> Self;
}

impl Power for f32 {
    fn epow(self, n: Self) -> Self {
        epowf(self, n)
    }
}

impl Power for f64 {
    fn epow(self, n: Self) -> Self {
        epow(self, n)
    }
}

pub trait ArcTan {
    /// Computes arc tangent
    fn eatan(self) -> Self;
}

impl ArcTan for f32 {
    fn eatan(self) -> Self {
        eatanf(self)
    }
}

impl ArcTan for f64 {
    fn eatan(self) -> Self {
        eatan(self)
    }
}

pub trait ArcSin {
    /// Computes arc sine
    fn easin(self) -> Self;
}

impl ArcSin for f32 {
    fn easin(self) -> Self {
        easinf(self)
    }
}

impl ArcSin for f64 {
    fn easin(self) -> Self {
        easin(self)
    }
}

pub trait ArcCos {
    /// Computes arc cosine
    fn eacos(self) -> Self;
}

impl ArcCos for f32 {
    fn eacos(self) -> Self {
        eacosf(self)
    }
}

impl ArcCos for f64 {
    fn eacos(self) -> Self {
        eacos(self)
    }
}

pub trait ArcTan2 {
    fn eatan2(self, x: Self) -> Self;
}

impl ArcTan2 for f32 {
    fn eatan2(self, x: f32) -> Self {
        eatan2f(self, x)
    }
}

impl ArcTan2 for f64 {
    fn eatan2(self, x: Self) -> Self {
        eatan2(self, x)
    }
}

pub trait CubeRoot {
    /// Computes cube root
    fn ecbrt(self) -> Self;
}

impl CubeRoot for f32 {
    fn ecbrt(self) -> Self {
        ecbrtf(self)
    }
}

impl CubeRoot for f64 {
    fn ecbrt(self) -> Self {
        ecbrt(self)
    }
}

pub trait Euclidean3DDistance {
    /// Computes euclidean 3D distance
    fn hypot3(self, y: Self, z: Self) -> Self;
}

impl Euclidean3DDistance for f32 {
    fn hypot3(self, y: Self, z: Self) -> Self {
        ehypot3f(self, y, z)
    }
}

impl Euclidean3DDistance for f64 {
    fn hypot3(self, y: Self, z: Self) -> Self {
        ehypot3(self, y, z)
    }
}

pub trait Euclidean2DDistance {
    /// Computes euclidean 2D distance
    fn ehypot(self, y: Self) -> Self;
}

impl Euclidean2DDistance for f32 {
    fn ehypot(self, y: Self) -> Self {
        ehypotf(self, y)
    }
}

impl Euclidean2DDistance for f64 {
    fn ehypot(self, y: Self) -> Self {
        ehypot(self, y)
    }
}

pub trait Euclidean4DDistance {
    /// Computes euclidean 4D distance
    fn hypot4(self, y: Self, z: Self, w: Self) -> Self;
}

impl Euclidean4DDistance for f32 {
    fn hypot4(self, y: Self, z: Self, w: Self) -> Self {
        ehypot4f(self, y, z, w)
    }
}

impl Euclidean4DDistance for f64 {
    fn hypot4(self, y: Self, z: Self, w: Self) -> Self {
        ehypot4(self, y, z, w)
    }
}

pub trait FusedMultiplyAdd {
    /// Computes `self * b + c`
    fn mla(self, b: Self, c: Self) -> Self;
}

impl FusedMultiplyAdd for f32 {
    fn mla(self, b: Self, c: Self) -> Self {
        mlaf(self, b, c)
    }
}

impl FusedMultiplyAdd for f64 {
    fn mla(self, b: Self, c: Self) -> Self {
        mlaf(self, b, c)
    }
}
