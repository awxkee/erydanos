use rand::Rng;
use rug::{Float, Integer};
use std::ops::{Add, Mul, Sub};

pub fn add_bits(k: f32, r_bits: u32) -> f32 {
    // Some targets violate Rust's assumption of IEEE semantics, e.g. by flushing
    // denormals to zero. This is in general unsound and unsupported, but here
    // we do our best to still produce the correct result on such targets.
    let bits = k.to_bits();
    if k.is_nan() || bits == f32::INFINITY.to_bits() {
        return k;
    }

    let abs = bits & !0x8000_0000;
    let next_bits = if abs == 0 {
        0x1
    } else if bits == abs {
        bits + r_bits
    } else {
        bits - r_bits
    };
    f32::from_bits(next_bits)
}

pub fn sub_bits(k: f32, r_bits: u32) -> f32 {
    // Some targets violate Rust's assumption of IEEE semantics, e.g. by flushing
    // denormals to zero. This is in general unsound and unsupported, but here
    // we do our best to still produce the correct result on such targets.
    let bits = k.to_bits();
    if k.is_nan() || bits == f32::INFINITY.to_bits() {
        return k;
    }

    let abs = bits & !0x8000_0000;
    let next_bits = if abs == 0 {
        0x1
    } else if bits == abs {
        bits - r_bits
    } else {
        bits + r_bits
    };
    f32::from_bits(next_bits)
}

pub fn add_bits_f64(k: f64, r_bits: u64) -> f64 {
    // Some targets violate Rust's assumption of IEEE semantics, e.g. by flushing
    // denormals to zero. This is in general unsound and unsupported, but here
    // we do our best to still produce the correct result on such targets.
    let bits = k.to_bits();
    if k.is_nan() || bits == f64::INFINITY.to_bits() {
        return k;
    }

    let abs = bits & !0x8000_0000;
    let next_bits = if abs == 0 {
        0x1
    } else if bits == abs {
        bits + r_bits
    } else {
        bits - r_bits
    };
    f64::from_bits(next_bits)
}

pub fn sub_bits_f64(k: f64, r_bits: u64) -> f64 {
    // Some targets violate Rust's assumption of IEEE semantics, e.g. by flushing
    // denormals to zero. This is in general unsound and unsupported, but here
    // we do our best to still produce the correct result on such targets.
    let bits = k.to_bits();
    if k.is_nan() || bits == f64::INFINITY.to_bits() {
        return k;
    }

    let abs = bits & !0x8000_0000;
    let next_bits = if abs == 0 {
        0x1
    } else if bits == abs {
        bits - r_bits
    } else {
        bits + r_bits
    };
    f64::from_bits(next_bits)
}

#[inline(always)]
pub fn random_coeff(d: f32) -> f32 {
    let mut rng = rand::thread_rng();

    let mut f1 = rug::Float::with_val(23, d);
    let scale = Float::with_val(f1.prec(), f32::EPSILON);
    let n2: u32 = rng.gen_range(0..=85500);
    let n1: u32 = rng.gen_range(0..=n2);
    let increment = Float::with_val(f1.prec(), scale).mul(Integer::from(n1));
    let goes_up = rng.gen_bool(1.0 / 2.0);
    if goes_up {
        f1 = f1.add(increment);
    } else {
        f1 = f1.sub(increment);
    }
    let new_value = f1.to_f32();
    return new_value;
}

pub fn random_coeff_f64(d: f64) -> f64 {
    let mut rng = rand::thread_rng();

    let mut f1 = rug::Float::with_val(53, d);
    let scale = Float::with_val(f1.prec(), f64::EPSILON);

    let n2: u64 = rng.gen_range(0..=1023);
    let n1: u64 = rng.gen_range(0..=n2);
    let increment = Float::with_val(f1.prec(), scale).mul(Integer::from(n1));
    let goes_up = rng.gen_bool(1.0 / 2.0);
    if goes_up {
        f1 = f1.add(increment);
    } else {
        f1 = f1.sub(increment);
    }
    let new_value = f1.to_f64();
    return new_value;
}
