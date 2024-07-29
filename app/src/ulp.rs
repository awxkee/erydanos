use libm::{fabs, fabsf};
// use rug::{Assign, Float};

// pub fn count_ulp_f64(d: f64, c: &Float) -> f64 {
//     let c2 = c.to_f64();
//
//     if (c2 == 0. || c2.is_subnormal()) && (d == 0. || d.is_subnormal()) {
//         return 0.;
//     }
//
//     if (c2 == 0.) && (d != 0.) {
//         return 10000.;
//     }
//
//     if c2.is_infinite() && d.is_infinite() {
//         return 0.;
//     }
//
//     let prec = c.prec();
//
//     let mut fry = Float::with_val(prec, d);
//
//     let mut frw = Float::new(prec);
//
//     let (_, e) = c.to_f64_exp();
//
//     frw.assign(Float::u_exp(1, e - 53_i32));
//
//     fry -= c;
//     fry /= &frw;
//     let u = fabs(fry.to_f64());
//
//     u
// }
//
// pub fn count_ulp(d: f32, c: &Float) -> f32 {
//     let c2 = c.to_f32();
//
//     if (c2 == 0. || c2.is_subnormal()) && (d == 0. || d.is_subnormal()) {
//         return 0.;
//     }
//
//     if (c2 == 0.) && (d != 0.) {
//         return 10000.;
//     }
//
//     if c2.is_infinite() && d.is_infinite() {
//         return 0.;
//     }
//
//     let prec = c.prec();
//
//     let mut fry = Float::with_val(prec, d);
//
//     let mut frw = Float::new(prec);
//
//     let (_, e) = c.to_f32_exp();
//
//     frw.assign(Float::u_exp(1, e - 24_i32));
//
//     fry -= c;
//     fry /= &frw;
//     let u = fabsf(fry.to_f32());
//
//     u
// }

pub fn count_ulp_f64(d: f64, c2: f64) -> f64 {
    if (c2 == 0. || c2.is_subnormal()) && (d == 0. || d.is_subnormal()) {
        return 0.;
    }

    if (c2 == 0.) && (d != 0.) {
        return 10000.;
    }

    if c2.is_infinite() && d.is_infinite() {
        return 0.;
    }
    let m = d.max(c2);
    let mmm = d.min(c2);
    return (m.to_bits() - mmm.to_bits()) as f64;
}

pub fn count_ulp(d: f32, c2: f32) -> f32 {
    if (c2 == 0. || c2.is_subnormal()) && (d == 0. || d.is_subnormal()) {
        return 0.;
    }

    if (c2 == 0.) && (d != 0.) {
        return 10000.;
    }

    if c2.is_infinite() && d.is_infinite() {
        return 0.;
    }
    let m = d.max(c2);
    let mmm = d.min(c2);
    return (m.to_bits() - mmm.to_bits()) as f32;
}
