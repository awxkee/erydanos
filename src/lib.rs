/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::abs::{eabs, eabsf};
use crate::acos::eacos;
use crate::acosf::eacosf;
use crate::asin::easin;
use crate::asinf::easinf;
use crate::atan::eatan;
use crate::atan2::eatan2;
use crate::atan2f::eatan2f;
use crate::atanf::eatanf;
use crate::cbrt::ecbrt;
use crate::cbrtf::ecbrtf;
use crate::cos::ecos;
use crate::cosf::ecosf;
use crate::exp::eexp;
use crate::expf::eexpf;
use crate::floor::{efloor, efloorf};
use crate::ln::eln;
use crate::lnf::elnf;
use crate::pow::epow;
use crate::powf::epowf;
use crate::sin::esin;
use crate::sinf::esinf;
use crate::sqrt::esqrt;
use crate::sqrtf::esqrtf;
use crate::tan::etan;
use crate::tanf::etanf;

pub mod abs;
pub mod acos;
pub mod acosf;
pub mod asin;
pub mod asinf;
pub mod atan;
pub mod atan2;
pub mod atan2f;
pub mod atanf;
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
pub mod fmax;
pub mod fmaxf;
pub mod fmin;
pub mod fminf;
pub mod generalf;
pub mod hypot;
pub mod hypotf;
pub mod ln;
pub mod lnf;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
pub mod neon;
pub mod pow;
pub mod powf;
pub mod sin;
pub mod sinf;
pub mod sqrt;
pub mod sqrtf;
pub mod tan;
pub mod tanf;
mod vector;

pub trait Sqrtf {
    fn esqrt(self) -> Self;
}

pub trait Cosine {
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
