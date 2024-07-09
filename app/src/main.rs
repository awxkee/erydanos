#![feature(float_next_up_down)]
#![feature(const_float_bits_conv)]

use std::arch::aarch64::*;
use std::ops::Shr;

use rug::Assign;

use crate::search_optimized_coeffs::search_coeffs_f32;
use crate::ulp::{count_ulp, count_ulp_f64};
use erydanos::abs::eabs;
use erydanos::{
    ArcCos, ArcSin, ArcTan, ArcTan2, Cosine, CubeRoot, Exponential, Logarithmic, Power, Sine,
    Tangent,
};

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
    let ag = (0.185f64).etan();
    let rg = rug::Float::tan(rug::Float::with_val(100, 0.1850f64));
    println!("approx {}, real {}", ag, rg.to_f64(),);
    println!("{}", count_ulp_f64(ag, &rg));
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

    for i in -200..200 {
        let scale = 0.005f32;
        let x = 1f32;
        let ap = (i as f32 * scale).eexp();

        let ax = rug::Float::with_val(100, i as f32 * scale);
        let rg = rug::Float::exp(ax);
        let lm = rg.to_f32();

        let ulp = count_ulp(ap, &rg) as f64;
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
                println!("ULP {} error {}", ulp, (i as f32 * scale));
            }
            max_ulp = ulp as f64;
        }

        // println!("value {}, app rempif {}, {}", i as f32 * scale, ap, lm,)
    }
    //
    // for i in -200..200 {
    //     let scale = 0.005f64;
    //     let counted = rug::Float::tan(rug::Float::with_val(100, i as f64 * scale));
    //     let ap = (i as f64 * scale).etan();
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
    //     println!(
    //         "value {}, ulp {}, app rempif {}, {}",
    //         i as f64 * scale,
    //         ulp,
    //         ap,
    //         lm,
    //     )
    // }
    println!("Worst ULP {}", max_ulp);
    // search_coeffs_f32();
    // search_coeffs_f64();
}
