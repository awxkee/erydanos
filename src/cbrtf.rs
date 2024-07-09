/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};
use std::ops::{Add, Div, Mul};

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::cbrtf::vcbrtq_f32;
use num_traits::AsPrimitive;

#[inline]
pub(crate) fn halley_cbrt<T: Copy + Mul<Output = T> + Div<Output = T> + Add<Output = T> + 'static>(
    x: T,
    a: T,
) -> T
where
    f32: AsPrimitive<T>,
{
    let tx = x * x * x;
    x * (tx + 2f32.as_() * a) / (2f32.as_() * tx + a)
}

const B1: u32 = 709958130;

fn do_cbrtf(x: f32) -> f32 {
    if x == 0. {
        return x;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }
    if x == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }

    let mut t: f32;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7fffffff;

    hx = hx / 3 + B1;
    ui &= 0x80000000;
    ui |= hx;

    t = f32::from_bits(ui);
    t = halley_cbrt(t, x);
    t = halley_cbrt(t, x);
    t
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline]
fn do_cbrtf_neon(d: f32) -> f32 {
    unsafe {
        let ld = vdupq_n_f32(d);
        vgetq_lane_f32::<0>(vcbrtq_f32(ld))
    }
}

/// Takes cube root from value *ULP 1.5*
#[inline]
pub fn ecbrtf(x: f32) -> f32 {
    let mut _dispatcher: fn(f32) -> f32 = do_cbrtf;
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _dispatcher = do_cbrtf_neon;
    }
    _dispatcher(x)
}
