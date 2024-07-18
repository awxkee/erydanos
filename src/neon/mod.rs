/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

mod acos;
mod acosf;
mod asin;
mod asinf;
mod atan2f;
mod atanf;
mod cbrt;
mod cbrtf;
mod ceil;
mod ceilf;
mod cos;
mod cosf;
mod exp;
mod expf;
mod float128;
mod floor;
mod floorf;
mod fmod;
mod fmodf;
mod general;
mod hypot;
mod hypot3;
mod hypot3f;
mod hypot4;
mod hypot4f;
mod hypotf;
mod ln;
mod lnf;
mod poly128;
mod pow;
mod powf;
mod sin;
mod sinf;
mod tan;
mod tanf;

pub use acos::vacosq_f64;
pub use acosf::vacosq_f32;
pub use asin::vasinq_f64;
pub use asinf::vasinq_f32;
pub use atan2f::vatan2q_f32;
pub use atanf::vatanq_f32;
pub use cbrt::vcbrtq_f64;
pub use cbrt::vcbrtq_fast_f64;
pub use cbrtf::vcbrtq_f32;
pub use cbrtf::vcbrtq_fast_f32;
pub use ceil::vceilq_f64;
pub use ceil::vceilq_ie_f64;
pub use ceilf::vceilq_f32;
pub use ceilf::vceilq_ie_f32;
pub use cos::vcosq_f64;
pub use cosf::vcosq_f32;
pub use exp::vexpq_f64;
pub use expf::vexpq_f32;
pub use expf::vexpq_fast_f32;
pub use float128::*;
pub use floor::vfloorq_f64;
pub use floor::vfloorq_ie_f64;
pub use floorf::vfloorq_f32;
pub use floorf::vfloorq_ie_f32;
pub use fmod::vfmodq_f64;
pub use fmodf::vfmodq_f32;
pub use general::vcopysignq_f32;
pub use general::vcopysignq_f64;
pub use general::vilogb2kq_f32;
pub use general::vilogb2kq_f64;
pub use general::visinfq_f32;
pub use general::visinfq_f64;
pub use general::visnanq_f32;
pub use general::visnanq_f64;
pub use general::visneginfq_f32;
pub use general::visneginfq_f64;
pub use general::visnotintegralq_f32;
pub use general::visnotintegralq_f64;
pub use general::vmlafq_f32;
pub use general::vmlafq_f64;
pub use general::vmlsfq_f32;
pub use general::vmlsfq_f64;
pub use general::vmul_s64;
pub use general::vmul_u64;
pub use general::vmulq_s64;
pub use general::vmulq_u64;
pub use general::vmulsignq_f64;
pub use general::vpow2ifq_s32;
pub use general::vpow2ifq_s64;
pub use general::vqshrn_n_u128;
pub use general::vsignbitq_f32;
pub use general::vsignbitq_f64;
pub use general::*;
pub use hypot::vhypotq_f64;
pub use hypot3::vhypot3q_f64;
pub use hypot3::vhypot3q_fast_f64;
pub use hypot3f::vhypot3q_f32;
pub use hypot3f::vhypot3q_fast_f32;
pub use hypot4::vhypot4q_f64;
pub use hypot4::vhypot4q_fast_f64;
pub use hypot4f::vhypot4q_f32;
pub use hypot4f::vhypot4q_fast_f32;
pub use hypotf::vhypotq_f32;
pub use hypotf::vhypotq_fast_f32;
pub use ln::vlnq_f64;
pub use lnf::vlnq_f32;
pub use lnf::vlnq_fast_f32;
pub use poly128::vshl_n_s128;
pub use poly128::vshl_n_u128;
pub use poly128::vshlq_n_s128;
pub use poly128::vshlq_n_u128;
pub use poly128::*;
pub use pow::vpowq_f64;
pub use powf::vpowq_f32;
pub use powf::vpowq_fast_f32;
pub use sin::vsinq_f64;
pub use sinf::vsinq_f32;
pub use tan::vtanq_f64;
pub use tanf::vtanq_f32;
