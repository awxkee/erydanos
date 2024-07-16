use crate::random_coeffs::{random_coeff, random_coeff_f64};
use crate::ulp::{count_ulp, count_ulp_f64};
use erydanos::asin::*;
use erydanos::asinf::*;
use erydanos::atanf::*;
use erydanos::exp::do_exp_coeff;

pub fn search_coeffs_f32() {
    let mut max_ulp: f64 = 0.;
    let mut initial_coeffs = vec![];
    initial_coeffs.push(6.4716694430273e-8);
    initial_coeffs.push(0.33923429007377);
    initial_coeffs.push(-2.4121886543858e-7);
    initial_coeffs.push(0.10458546797000);
    initial_coeffs.push(3.8563211233698e-7);
    initial_coeffs.push(0.10445851491159);
    initial_coeffs.push(-2.2156168553412e-7);

    let mut best_match_coeeffs: Vec<f32> = initial_coeffs.clone().iter().map(|&x| x).collect();
    let mut best_ulp = f64::MAX;
    let mut best_value = 0f32;

    for _ in 0..25000 {
        let new_coeffs: Vec<f32> = initial_coeffs
            .clone()
            .iter()
            .map(|&x| random_coeff(x))
            .collect();

        let mut best_val = 0f32;
        max_ulp = 0.;

        for i in -200..200 {
            let scale = 0.005f32;

            // let counted = rug::Float::tan(rug::Float::with_val(100, i as f32 * scale));
            // let ap = do_tan_coeffs(i as f32 * scale, &new_coeffs);
            // let lm = counted.to_f32();
            //
            // let ulp = count_ulp(ap, &counted) as f64;
            // if ulp > max_ulp {
            //     max_ulp = ulp;
            //     best_val = i as f32 * scale;
            // }
        }

        if max_ulp < best_ulp {
            best_match_coeeffs = new_coeffs.clone().iter().map(|&x| x).collect();
            best_ulp = max_ulp;
            best_value = best_val;
        }
    }
    println!("Best value {}", best_value);
    println!("Best ULP {}", best_ulp);
    println!("Best Coefficients {:?}", best_match_coeeffs);
}

pub fn search_coeffs_f64() {
    let mut max_ulp: f64 = 0.;
    let mut initial_coeffs = vec![];
    // initial_coeffs.push(EXP_POLY_2_D);
    // initial_coeffs.push(EXP_POLY_3_D);
    // initial_coeffs.push(EXP_POLY_4_D);
    // initial_coeffs.push(EXP_POLY_5_D);
    // initial_coeffs.push(EXP_POLY_6_D);

    let mut best_match_coeeffs: Vec<f64> = initial_coeffs.clone().iter().map(|&x| x).collect();
    let mut best_ulp = f64::MAX;
    let mut best_value = 0.;

    for _ in 0..10000 {
        let new_coeffs: Vec<f64> = initial_coeffs
            .clone()
            .iter()
            .map(|&x| random_coeff_f64(x))
            .collect();

        let mut best_val = 0.;
        max_ulp = 0.;

        for i in -2000..2000 {
            let scale = 0.005f64;

            let counted = rug::Float::exp(rug::Float::with_val(100, i as f64 * scale));
            let ap = do_exp_coeff(i as f64 * scale, &new_coeffs);

            let ulp = count_ulp_f64(ap, &counted) as f64;
            if ulp > max_ulp {
                max_ulp = ulp;
                best_val = i as f64 * scale;
            }
        }

        if max_ulp < best_ulp {
            best_match_coeeffs = new_coeffs.clone().iter().map(|&x| x).collect();
            best_ulp = max_ulp;
            best_value = best_val;
        }
    }
    println!("Best value {}", best_value);
    println!("Best ULP {}", best_ulp);
    println!("Best Coefficients {:?}", best_match_coeeffs);
}
