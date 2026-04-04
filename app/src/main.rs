use std::fs;
use std::ops::{Add, Mul, Shr, Sub};
use libm::{log, log2};
use rug::Float;
use rug::float::Round;
// use rug::Assign;

use erydanos::{
    eabs, ecos, ecosf, eexp, eln, epow, esin, esinf, ArcCos, ArcSin, ArcTan, ArcTan2, Cosine,
    CubeRoot, Exponential, Logarithmic, Power, Sine, Tangent,
};

use crate::ulp::{count_ulp, count_ulp_d, count_ulp_f64};

mod random_coeffs;
mod search_optimized_coeffs;
mod ulp;

fn factorial(n: u64) -> f64 {
    (1..=n).map(|x| x as f64).product()
}

fn next_representable(value: f32) -> f32 {
    if value.is_nan() || value.is_infinite() {
        return value;
    }

    let bits = value.to_bits();
    let next_bits = if value > 0.0 {
        bits - 1
    } else if value < 0.0 {
        bits + 1
    } else {
        1 // the smallest positive subnormal number
    };

    f32::from_bits(next_bits)
}

// fn pow2f(x: f32) -> f32 {
//     let dx = 0.215596346446f32 * x;
// }

fn closest_err(value: f64, fun: fn(f64, f64) -> f64, coeff: f64, approx: f64) -> f64 {
    let mut closest_coeff = coeff;
    let mut error = f64::MAX;
    let mut prev = coeff;
    for i in 0..1_000_000_000u64 {
        // let new_coeff = if prev < 0f64 {
        //     prev.next_down()
        // } else {
        //     prev.next_up()
        // };
        let new_coeff = i as f64 * 0.0000000000001f64 + coeff;
        prev = new_coeff;
        let value = fun(value, new_coeff);
        let new_error = (value - approx).abs();
        if new_error < error {
            error = new_error;
            closest_coeff = new_coeff;
        }
    }

    println!("latest coeff {}", prev);

    closest_coeff
}

fn split_i64_to_i32_parts(x: i64) -> (i32, i32) {
    let low = (x & 0xFFFFFFFF) as i32;
    let high = (x >> 32) as i32;
    (low, high)
}

fn multiply_i32_to_i64(a: i32, b: i32) -> i32 {
    (a as i32) * (b as i32)
}

fn combine_parts(low_low: i64, low_high: i64, high_low: i64, high_high: i64) -> i64 {
    // Combine the parts, managing the positions and overflow
    (low_low + low_high) + (high_high + high_low) << 32
}

#[inline(always)]
fn multiply_u64(a: u64, b: u64) -> (u64, u64) {
    let a_low = a & 0xFFFFFFFF;
    let a_high = a >> 32;
    let b_low = b & 0xFFFFFFFF;
    let b_high = b >> 32;

    let low_low = a_low.wrapping_mul(b_low);
    let low_high = a_low.wrapping_mul(b_high);
    let high_low = a_high.wrapping_mul(b_low);
    let high_high = a_high.wrapping_mul(b_high);

    let mid1 = (low_low >> 32)
        .wrapping_add(low_high & 0xFFFFFFFF)
        .wrapping_add(high_low & 0xFFFFFFFF);
    let mid2 = (mid1 >> 32)
        .wrapping_add(low_high >> 32)
        .wrapping_add(high_low >> 32)
        .wrapping_add(high_high);

    let result_low = (low_low & 0xFFFFFFFF).wrapping_add(mid1 << 32);
    let result_high = mid2;

    (result_low, result_high)
}

#[inline(always)]
fn lhs_u128(low: u64, high: u64, shift: i64) -> (u64, u64) {
    if (shift < 0) {
        panic!("Shift count cannot be negative");
    }
    let (lo, mut hi);
    if (shift >= 64) {
        lo = 0;
        hi = low << (shift - 64);
    } else {
        lo = low << shift;
        hi = high << shift;

        // Handle the overflow from lower to upper part
        hi |= low >> (64 - shift);
    }

    return (lo, hi);
}

#[inline(always)]
fn lhs_s128(low: i64, high: i64, shift: i64) -> (i64, i64) {
    if (shift < 0) {
        panic!("Shift count cannot be negative");
    }
    let (lo, mut hi);
    if (shift >= 64) {
        lo = 0;
        hi = low << (shift - 64);
    } else {
        lo = low << shift;
        hi = high << shift;

        // Handle the overflow from lower to upper part
        hi |= low >> (64 - shift);
    }

    return (lo, hi);
}

fn add_u128(a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
    // Add the lower parts
    let rs_lo = a.0.wrapping_add(b.0);

    // Check for carry from the lower part addition
    let carry = rs_lo < a.0;

    // Add the upper parts along with the carry
    let rs_hi =
        a.1.wrapping_add(b.1.wrapping_add(if carry { 1 } else { 0 }));

    return (rs_lo, rs_hi);
}

fn add_s128(a: (i64, i64), b: (i64, i64)) -> (i64, i64) {
    // Add the lower parts
    let rs_lo = a.0.wrapping_add(b.0);

    // Check for carry from the lower part addition
    let carry = rs_lo < a.0;

    // Add the upper parts along with the carry
    let rs_hi =
        a.1.wrapping_add(b.1.wrapping_add(if carry { 1 } else { 0 }));

    return (rs_lo, rs_hi);
}

#[inline(always)]
#[no_mangle]
fn multiply_ui64(lhs: u64, rhs: u64) -> (u64, u64) {
    let a_high = lhs >> 32;
    let a_low = lhs & 0xffffffff;
    let b_high = rhs >> 32;
    let b_low = rhs & 0xffffffff;

    let low1 = a_low * b_low;
    let low2 = a_low * b_high;
    let low3 = a_high * b_low;
    let high = a_high * b_high;

    let mut hi = high;
    let mut lo = low1;
    let carry1 = lhs_u128(low3, 0, 32);
    let carry2 = lhs_u128(low2, 0, 32);
    let mut result = (lo, hi);
    result = add_u128(result, carry1);
    result = add_u128(result, carry2);
    return result;
}

#[inline(always)]
#[no_mangle]
fn multiply_i64(lhs: i64, rhs: i64) -> (i64, i64) {
    let a_high = lhs >> 32;
    let a_low = lhs & 0xffffffff;
    let b_high = rhs >> 32;
    let b_low = rhs & 0xffffffff;

    let low1 = a_low * b_low;
    let low2 = a_low * b_high;
    let low3 = a_high * b_low;
    let high = a_high * b_high;

    let mut hi = high;
    let mut lo = low1;
    let carry1 = lhs_s128(low3, 0, 32);
    let carry2 = lhs_s128(low2, 0, 32);
    let mut result = (lo, hi);
    result = add_s128(result, carry1);
    result = add_s128(result, carry2);
    return result;
}

fn add_with_overflow_detection(a: i64, b: i64) -> (i64, bool) {
    let sum = a.wrapping_add(b);
    let overflow = (!(a ^ b) & (a ^ sum)) < 0;
    (sum, overflow)
}


fn split_to_double_double(input: &str) -> (f64, f64) {
    // Step 1: High-precision number with extra bits (e.g., 128 bits)
    let valid = Float::parse(input);
    let mut x = Float::with_val(128, valid.unwrap());

    // Step 2: Round down to nearest f64
    let hi = x.to_f64();
    // let hi_trunc = f64::from_bits(hi.to_bits() & 0x_ffff_ffff_f800_0000);
    // let hi_float = Float::with_val(64, hi_trunc);
    let hi_float = Float::with_val(150, hi);

    // Step 3: Subtract to get low part
    x -= &hi_float;
    let lo = x.to_f64();

    (hi, lo)
}

fn to_hex_u64(input: &str) -> u64 {
    // Step 1: High-precision number with extra bits (e.g., 128 bits)
    let valid = Float::parse(input);
    let mut x = Float::with_val(128, valid.unwrap());
    x.to_f64_round(Round::Nearest).to_bits()
}

fn split_to_f_f(input: &str) -> (f32, f32) {
    // Step 1: High-precision number with extra bits (e.g., 128 bits)
    let valid = Float::parse(input);
    let mut x = Float::with_val(128, valid.unwrap());

    // Step 2: Round down to nearest f64
    let hi = x.to_f32();
    let hi_float = Float::with_val(128, hi);

    // Step 3: Subtract to get low part
    x -= &hi_float;
    let lo = x.to_f32();

    (hi, lo)
}


fn split_to_double_double_f(input: &Float) -> (f64, f64) {
    // Step 1: High-precision number with extra bits (e.g., 128 bits)
    // Step 2: Round down to nearest f64
    let hi = input.to_f64();
    let hi_float = Float::with_val(input.prec(), hi);
    // let hi_trunc = f64::from_bits(hi.to_bits() & 0x_ffff_ffff_f800_0000);
    // let hi_float = Float::with_val(64, hi_trunc);
    let mut x = input.clone();
    // Step 3: Subtract to get low part
    x -= &hi_float;
    let lo = x.to_f64();

    (hi, lo)
}

fn split_to_double_double_f_d(input: &Float) -> (f64, Float) {
    // Step 1: High-precision number with extra bits (e.g., 128 bits)
    // Step 2: Round down to nearest f64
    let hi = input.to_f64();
    let hi_float = Float::with_val(128, hi);
    // let hi_trunc = f64::from_bits(hi.to_bits() & 0x_ffff_ffff_f800_0000);
    // let hi_float = Float::with_val(150, hi_trunc);
    let mut x = input.clone();
    // Step 3: Subtract to get low part
    x -= &hi_float;

    (hi, x)
}

fn get_log2_1() -> Float {
    let v = Float::parse("0.69314718055994530941723212145817656807550013436025525412068000949339362196");
    let mut log2_t = Float::with_val(150, v.unwrap());

    let bits = log2_t.to_f64().to_bits() & ((0xffff_ffff_ffff_ffff >> 10) << 10);
    let x = Float::with_val(150, f64::from_bits(bits));
    
    // let valid = Float::parse("0.693147180559945");
    // let x = Float::with_val(150, valid.unwrap());
    log2_t -= &x;
    log2_t
}

fn split_to_float_float(input: &Float) -> (f32, f32) {
    // Step 1: High-precision number with extra bits (e.g., 128 bits)
    // Step 2: Round down to nearest f64
    let hi = input.to_f32();
    let hi_float = Float::with_val(64, hi);

    let mut x = input.clone();
    // Step 3: Subtract to get low part
    x -= &hi_float;
    let lo = x.to_f32();

    (hi, lo)
}

#[derive(Debug, Clone, Copy)]
struct Exp2Entry<T> {
    factor: T,
    eps: T,
}

/// Generate a lookup table with TBLSIZE entries.
/// We map raw index [0, TBLSIZE) to signed integers in [-TBLSIZE/2, TBLSIZE/2 - 1].
fn generate_exp2_table<const TBLSIZE: usize>() -> [Exp2Entry<f64>; TBLSIZE] {
    let prec = 150u32; // high internal precision
    let mut table: [Exp2Entry<f64>; TBLSIZE] = [Exp2Entry { factor: 0.0, eps: 0.0 }; TBLSIZE];

    // For each raw index 0..256, compute signed index and table entry.
    for raw_i in 0..TBLSIZE {
        // Map: 0 -> -128, ..., 255 -> 127.
        // Compute fraction = i_signed / TBLSIZE.
        let mut fraction = Float::with_val(150, -0.5 + raw_i as f64 / (TBLSIZE as f64));
        
        // Compute high-precision exp2(fraction).
        let eexp0 = fraction.clone().exp().to_f64();

        let e1 = Float::with_val(150, eexp0);

        // Now, we want to choose eps such that:
        //    factor = 2^(fraction + eps)
        // equals the high-precision value rounded to f64.
        // One can define eps = log2(val_hp) - fraction.
        let log2_val = e1.clone().ln();
        let eps_hp = log2_val.clone().sub(fraction.clone());
        let eps = eps_hp.to_f64();

        // The factor: we want 2^(fraction + eps) computed in high precision,
        // then rounded to f64.
        let factor_hp = Float::with_val(prec, (fraction.clone() + eps_hp.clone()).exp());
        let factor = factor_hp.to_f64();

        table[raw_i] = Exp2Entry { factor, eps };
    }
    table
}

fn generate_log2_table<const TBLSIZE: usize>() -> [Exp2Entry<f64>; TBLSIZE] {
    let prec = 150u32; // high internal precision
    let mut table: [Exp2Entry<f64>; TBLSIZE] = [Exp2Entry { factor: 0.0, eps: 0.0 }; TBLSIZE];

    // For each raw index 0..256, compute signed index and table entry.
    for raw_i in 0..TBLSIZE {
        // Map: 0 -> -128, ..., 255 -> 127.
        // Compute fraction = i_signed / TBLSIZE.
        let mut fraction = Float::with_val(150, -0.5 + raw_i as f64 / (TBLSIZE as f64));

        // Compute high-precision exp2(fraction).
        let eexp0 = fraction.clone().exp().to_f64();

        let e1 = Float::with_val(150, eexp0);

        // Now, we want to choose eps such that:
        //    factor = 2^(fraction + eps)
        // equals the high-precision value rounded to f64.
        // One can define eps = log2(val_hp) - fraction.
        let log2_val = e1.clone().ln();
        let eps_hp = log2_val.clone().sub(fraction.clone());
        let eps = eps_hp.to_f64();

        // The factor: we want 2^(fraction + eps) computed in high precision,
        // then rounded to f64.
        let factor_hp = Float::with_val(prec, (fraction.clone() + eps_hp.clone()).exp());
        let factor = factor_hp.to_f64();

        table[raw_i] = Exp2Entry { factor, eps };
    }
    table
}

fn generate_exp2_table_f32<const TBLSIZE: usize>() -> [Exp2Entry<f32>; TBLSIZE] {
    let prec = 150u32; // high internal precision
    let mut table: [Exp2Entry<f32>; TBLSIZE] = [Exp2Entry { factor: 0.0, eps: 0.0 }; TBLSIZE];

    // For each raw index 0..256, compute signed index and table entry.
    for raw_i in 0..TBLSIZE {
        // Map: 0 -> -128, ..., 255 -> 127.
        // Compute fraction = i_signed / TBLSIZE.
        let mut fraction = Float::with_val(150, -0.5 + raw_i as f64 / (TBLSIZE as f64));

        // Compute high-precision exp2(fraction).
        let eexp0 = fraction.clone().exp2().to_f32();

        let e1 = Float::with_val(150, eexp0);

        // Now, we want to choose eps such that:
        //    factor = 2^(fraction + eps)
        // equals the high-precision value rounded to f64.
        // One can define eps = log2(val_hp) - fraction.
        let log2_val = e1.clone().log2();
        let eps_hp = log2_val.clone().sub(fraction.clone());
        let eps = eps_hp.to_f32();

        // The factor: we want 2^(fraction + eps) computed in high precision,
        // then rounded to f64.
        let factor_hp = Float::with_val(prec, (fraction.clone() + eps_hp.clone()).exp2());
        let factor = factor_hp.to_f32();

        table[raw_i] = Exp2Entry { factor, eps };
    }
    table
}

fn main() {
    // for i in -200..200 {
    //     let scale = 0.001f32;
    //     println!("value {}, real {}, k {}, app {}",scale * i as f32 ,2f32.powf(scale * i as f32), f32::exp2(scale * i as f32), exp2_approx(scale as f64 * i as f64))
    // }
    // println!("{}", closest_err(0.82f32, ecosft, -0.00002480158730158730158730158730f32, 0.6822212072f32));
    // println!("{}", closest_err(1.95f64, do_exp_t, 0.009618129107f64,
    //                            7.0286875805892933342908819335643795001448882776914963128865953514f64));
    /// original value 1.95, app rempif 7.02867214722499, 7.028687580589293
    /// MATHEMATICA 7.02868758058929
    let x = 2.0f32;
    let y = 32f32;
    let z = 12f32;
    println!("{:?}", 0f32.exp2());
    println!("{:?}", 5f32.exp2());
    println!("{:?}", esinf(0.5f32));
    println!("{:?}", multiply_ui64(u64::MAX, 2));
    println!("{}", u64::MAX as i128 * 2);
    let product = multiply_ui64((-4i64) as u64, (-2i64) as u64);
    println!(
        "sign {}, {}, product {}",
        product.0 as i64,
        product.1 as i64,
        product.0 as i128 | ((product.1 as i128) << 64)
    );
    println!("{:?}", multiply_u64(i64::MAX as u64, (-2i64) as u64));
    println!("{}", product.0 as i128 | ((product.1 as i128) << 64));
    
    println!("HEX {:#x}", to_hex_u64("0.14797756417918789262344603230303619056940078735351"));

    println!("{:?}", split_to_double_double("2.88539008177792681471984936200378427485329190830597186827089881386221843836"));
    
    // let mut f = Float::with_val(128, 512f64);
    // f = f.ln();
    // f = f.recip();
    // 
    // println!("F {:?}", split_to_double_double_f(&f));
    let k = f64::from_bits(0x3d39880000000000);

    println!("` {}", k);
    
    let rz = generate_exp2_table::<256>();
    println!("`1 {}", rz[0].eps);

    let hex_repr = rz
        .iter()
        .map(|&a| (a.factor.to_bits(), a.eps.to_bits()))
        .map(|(a0, a1)| format!("(0x{a0:08X}, 0x{a1:08X})"))
        .collect::<Vec<_>>()
        .join(",");

    fs::write("./bs.rs", format!("[{}]", hex_repr)).unwrap();
    
    /*let v1 = get_log2_1();
    println!("ln2_1({})", v1.to_f64_round(Round::Nearest));
    // 
    let mut lookup: [u64; 256] = [0; 256];
    for (i, loc) in lookup.iter_mut().enumerate() {
        
        let mut v = -0.5 + i as f64 / 256.0;
        
        // let n_v = (i as i64 - 128) & (256i64 - 1);
        // let n = Float::with_val(150, n_v);
        // let dely = (n.clone() * Float::with_val(150, v1.clone())).to_f64_round(Round::Nearest);
        // println!("{}", dely);
        // v += dely;
        let pe = Float::with_val(150, v).exp2();
        let splat = split_to_double_double_f(&pe);
        // *loc  = pe.to_f64_round(Round::Nearest).to_bits(); 
        *loc  = splat.0.to_bits();
        // println!("{}", i as f64 / 64.0);
    }
    
    let hex_repr = lookup
        .iter()
        .map(|a0| format!("0x{a0:08X}"))
        .collect::<Vec<_>>()
        .join(",");

    fs::write("./bs.rs", format!("[{}]", hex_repr)).unwrap();*/
    
    // let hex_repr = lookup
    //     .iter()
    //     .map(|(a, b)| format!("(0x{a:08X}, 0x{b:08X})"))
    //     .collect::<Vec<_>>()
    //     .join(",");

    // fs::write("./bs.rs", format!("[{}]", hex_repr)).unwrap();
    
    // fs::write("./bs.rs", format!("{:?}", lookup)).unwrap();
    
    // println!("{:?}", lookup);

    // println!(
    //     " bits diff {}",
    //     rg.to_f32().to_bits().max(ag.to_bits()) - rg.to_f32().to_bits().min(ag.to_bits())
    // );

    // unsafe {
    //     let set = [u64::MAX, 20u64];
    //     println!("{}", u64::MAX);
    //     println!(">> 32, {}", u64::MAX >> 32);
    //     let v1 = vdupq_n_u64(u64::MAX);
    //     let set2 = [2, 40u64];
    //     let v2 = vld1q_u64(set2.as_ptr());
    //     let mulled = vmull_u64(v1, v2);
    //     let first: u128 = vgetq_lane_u64::<0>(mulled.0) as u128 | ( (vgetq_lane_u64::<0>(mulled.1) as u128).shr(64) ) ;
    //     let second: u128 = vgetq_lane_u64::<1>(mulled.0) as u128 | ( (vgetq_lane_u64::<1>(mulled.1) as u128).shr(64) ) ;
    //     println!("First {}", first);
    //     println!("Second {}", second);
    //
    //     let shifted = vqshrn_n_u128::<1>(mulled);
    //     println!("First Divided {}, max /2  {}", vgetq_lane_u64::<0>(shifted), (u64::MAX / 2));
    //     println!("Second Divided {}", vgetq_lane_u64::<1>(shifted));
    // }

    // println!("approx {}, real {}", eatan(1.09f64), (1.09f64).atan());
    // // original value 1.58, app rempif 4.854955802915181, 4.854955811237434
    let mut cumulative_error = 0f64;

    let mut max_ulp: f32 = 0.;

    for i in 0..15000 {
        let scale = 0.000003f32;
        let x = 1f32;
        let ap = (i as f32 * scale).eatan();

        // let ax = rug::Float::with_val(100, i as f32 * scale);
        // let rg = rug::Float::exp(ax);
        // let lm = rg.to_f32();
        let rg = (i as f32 * scale).atan();
        let ulp = count_ulp(ap, rg) as f32;
        /*  if ulp > 1. {
            println!(
                "ULP {} error {}, approx {}, expected {}",
                ulp,
                (i as f32 * scale),
                ap,
                lm
            );
        }*/
        if ulp > max_ulp {
            if max_ulp > 10. {
                println!("ULP {} error {}", ulp, i as f32 * scale);
            }
            max_ulp = ulp as f32;
        }

        // println!("value {}, app rempif {}, {}", i as f32 * scale, ap, lm,)
    }
    //
    // for i in -3000..3000 {
    //     let scale = 0.005f64;
    //     // let counted = rug::Float::exp(rug::Float::with_val(100, i as f64 * scale));
    //     let coun
    //     let ap = (i as f64 * scale).eexp();
    //     let lm = counted.to_f64();
    //     if !ap.is_nan() {
    //         let diff = eabs(eabs(ap) - eabs(lm));
    //         cumulative_error += diff;
    //     }
    //
    //     let ulp = count_ulp_f64(ap, &counted);
    //     if ulp > max_ulp {
    //         if max_ulp > 1. {
    //             println!("ULP error {} for value {}", ulp, (i as f64 * scale));
    //         }
    //         max_ulp = ulp;
    //     }
    //
    //     // println!(
    //     //     "value {}, ulp {}, app rempif {}, {}",
    //     //     i as f64 * scale,
    //     //     ulp,
    //     //     ap,
    //     //     lm,
    //     // )
    // }
    println!("Worst ULP {}", max_ulp);
    // search_coeffs_f32();
    // search_coeffs_f64();
}
