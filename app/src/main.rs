use std::ops::{Add, Mul, Shr};

use rug::Assign;

use erydanos::{
    eabs, eexp, eln, epow, esin, ArcCos, ArcSin, ArcTan, ArcTan2, Cosine, CubeRoot, Exponential,
    Logarithmic, Power, Sine, Tangent,
};

use crate::ulp::count_ulp_f64;

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
    let ag = esin(-2.70752239);
    let rg = rug::Float::sin(rug::Float::with_val(53, -2.70752239));
    // println!("{:?}", multiply_ui64(2, 4));
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

    let mut max_ulp: f64 = 0.;

    // for i in -200..200 {
    //     let scale = 0.005f32;
    //     let x = 1f32;
    //     let ap = (i as f32 * scale).eexp();
    //
    //     let ax = rug::Float::with_val(100, i as f32 * scale);
    //     let rg = rug::Float::exp(ax);
    //     let lm = rg.to_f32();
    //
    //     let ulp = count_ulp(ap, &rg) as f64;
    //     /*  if ulp > 1. {
    //         println!(
    //             "ULP {} error {}, approx {}, expected {}",
    //             ulp,
    //             (i as f32 * scale),
    //             ap,
    //             lm
    //         );
    //     }*/
    //     if ulp > max_ulp {
    //         if max_ulp > 10. {
    //             println!("ULP {} error {}", ulp, (i as f32 * scale));
    //         }
    //         max_ulp = ulp as f64;
    //     }
    //
    //     // println!("value {}, app rempif {}, {}", i as f32 * scale, ap, lm,)
    // }
    //
    for i in -3000..3000 {
        let scale = 0.005f64;
        let counted = rug::Float::exp(rug::Float::with_val(100, i as f64 * scale));
        let ap = (i as f64 * scale).eexp();
        let lm = counted.to_f64();
        if !ap.is_nan() {
            let diff = eabs(eabs(ap) - eabs(lm));
            cumulative_error += diff;
        }

        let ulp = count_ulp_f64(ap, &counted);
        if ulp > max_ulp {
            if max_ulp > 1. {
                println!("ULP error {} for value {}", ulp, (i as f64 * scale));
            }
            max_ulp = ulp;
        }

        // println!(
        //     "value {}, ulp {}, app rempif {}, {}",
        //     i as f64 * scale,
        //     ulp,
        //     ap,
        //     lm,
        // )
    }
    println!("Worst ULP {}", max_ulp);
    // search_coeffs_f32();
    // search_coeffs_f64();
}
