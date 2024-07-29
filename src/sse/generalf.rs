/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
/// Founds n in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn _mm_ilogb2kq_ps(d: __m128) -> __m128i {
    _mm_sub_epi32(
        _mm_and_si128(
            _mm_srli_epi32::<23>(_mm_castps_si128(d)),
            _mm_set1_epi32(0xff),
        ),
        _mm_set1_epi32(0x7f),
    )
}

#[inline(always)]
/// Founds a in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn _mm_ldexp3kq_ps(x: __m128, n: __m128i) -> __m128 {
    _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(x), _mm_slli_epi32::<23>(n)))
}

#[cfg(not(target_feature = "fma"))]
#[inline]
/// Computes b*c + a using fma when available
pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    return _mm_add_ps(_mm_mul_ps(b, c), a);
}

#[cfg(target_feature = "fma")]
#[inline]
/// Computes b*c + a using fma when available
pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    return _mm_fmadd_ps(b, c, a);
}

#[inline(always)]
/// Computes a*b + c
pub unsafe fn _mm_mlaf_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_prefer_fma_ps(c, b, a)
}

#[inline(always)]
/// If mask then `true_vals` otherwise `false_val`
pub unsafe fn _mm_select_ps(mask: __m128, true_vals: __m128, false_vals: __m128) -> __m128 {
    _mm_blendv_ps(false_vals, true_vals, mask)
}

#[inline(always)]
/// If mask then `true_vals` otherwise `false_val`
pub unsafe fn _mm_selecti_ps(mask: __m128i, true_vals: __m128, false_vals: __m128) -> __m128 {
    _mm_blendv_ps(false_vals, true_vals, _mm_castsi128_ps(mask))
}

#[inline(always)]
/// Returns flag value is Infinity
pub unsafe fn _mm_isinf_ps(d: __m128) -> __m128 {
    return _mm_cmpeq_ps(_mm_abs_ps(d), _mm_set1_ps(f32::INFINITY));
}

#[inline(always)]
/// Returns flag value is Neg Infinity
pub unsafe fn _mm_isneginf_ps(d: __m128) -> __m128 {
    return _mm_cmpeq_ps(d, _mm_set1_ps(f32::NEG_INFINITY));
}

#[inline(always)]
/// Returns flag value is zero
pub unsafe fn _mm_eqzero_ps(d: __m128) -> __m128 {
    return _mm_cmpeq_ps(d, _mm_set1_ps(0.));
}

#[inline(always)]
/// Returns flag value is lower than zero
pub unsafe fn _mm_ltzero_ps(d: __m128) -> __m128 {
    return _mm_cmplt_ps(d, _mm_set1_ps(0.));
}

#[inline(always)]
/// Returns true flag if value is NaN
pub unsafe fn _mm_isnan_ps(d: __m128) -> __m128 {
    return _mm_cmpneq_ps(d, d);
}

#[inline(always)]
/// Modulus operator for f32
pub unsafe fn _mm_abs_ps(f: __m128) -> __m128 {
    return _mm_castsi128_ps(_mm_andnot_si128(
        _mm_castps_si128(_mm_set1_ps(-0.0f32)),
        _mm_castps_si128(f),
    ));
}

#[inline(always)]
/// Negates value
pub unsafe fn _mm_neg_ps(f: __m128) -> __m128 {
    _mm_sub_ps(_mm_set1_ps(0.), f)
}

#[inline(always)]
/// Rounds and takes integral part from float
pub unsafe fn _mm_rint_ps(f: __m128) -> __m128i {
    const ROUNDING_FLAG: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    let k = _mm_round_ps::<ROUNDING_FLAG>(f);
    _mm_cvtps_epi32(k)
}

#[inline(always)]
/// Computes 2^n in f32 form for signed 32 bits integers, returns f32 in bits
pub unsafe fn _mm_pow2if_epi32(n: __m128i) -> __m128i {
    let j = _mm_slli_epi32::<23>(_mm_add_epi32(n, _mm_set1_epi32(0x7f)));
    j
}

#[inline(always)]
/// Copies sign from `y` to `x`
pub unsafe fn _mm_copysign_ps(x: __m128, y: __m128) -> __m128 {
    _mm_castsi128_ps(_mm_xor_si128(
        _mm_andnot_si128(_mm_castps_si128(_mm_set1_ps(-0.0f32)), _mm_castps_si128(x)),
        _mm_and_si128(_mm_castps_si128(_mm_set1_ps(-0.0f32)), _mm_castps_si128(y)),
    ))
}

#[inline(always)]
/// Checks if arguments is integral value
pub unsafe fn _mm_isintegral_ps(d: __m128) -> __m128 {
    return _mm_cmpeq_ps(d, _mm_floor_ps(d));
}

#[inline(always)]
/// Checks if arguments is not integral value
pub unsafe fn _mm_isnotintegral_ps(d: __m128) -> __m128 {
    return _mm_cmpneq_ps(d, _mm_floor_ps(d));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_if_inf() {
        unsafe {
            // Test regular
            let value = _mm_set1_ps(23f32);
            let comparison = _mm_isinf_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_eq!(flag, 0);
        }

        unsafe {
            // Test negative regular
            let value = _mm_set1_ps(-23f32);
            let comparison = _mm_isinf_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_eq!(flag, 0);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_ps(f32::INFINITY);
            let comparison = _mm_isinf_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_ne!(flag, 0);
        }
    }

    #[test]
    fn test_if_neg_inf() {
        unsafe {
            // Test regular
            let value = _mm_set1_ps(23f32);
            let comparison = _mm_isneginf_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_eq!(flag, 0);
        }

        unsafe {
            // Test negative regular
            let value = _mm_set1_ps(-23f32);
            let comparison = _mm_isneginf_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_eq!(flag, 0);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_ps(f32::INFINITY);
            let comparison = _mm_isneginf_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_eq!(flag, 0);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_ps(f32::NEG_INFINITY);
            let comparison = _mm_isneginf_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_ne!(flag, 0);
        }
    }

    #[test]
    fn test_copy_sign() {
        unsafe {
            let value = _mm_set1_ps(23f32);
            let other = _mm_set1_ps(-2f32);
            let comparison = _mm_copysign_ps(value, other);
            let flag = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag, -23f32);
        }

        unsafe {
            let value = _mm_set1_ps(23f32);
            let other = _mm_set1_ps(2f32);
            let comparison = _mm_copysign_ps(value, other);
            let flag = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag, 23f32);
        }
    }

    #[test]
    fn test_abs() {
        unsafe {
            // Test regular
            let value = _mm_set1_ps(23f32);
            let comparison = _mm_abs_ps(value);
            let flag = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag, 23f32);
        }

        unsafe {
            // Test negative regular
            let value = _mm_set1_ps(-23f32);
            let comparison = _mm_abs_ps(value);
            let flag = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag, 23f32);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_ps(f32::NEG_INFINITY);
            let comparison = _mm_abs_ps(value);
            let flag = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag.is_infinite(), true);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_ps(f32::INFINITY);
            let comparison = _mm_abs_ps(value);
            let flag = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag.is_infinite(), true);
        }

        unsafe {
            // Test NaN
            let value = _mm_set1_ps(f32::NAN);
            let comparison = _mm_abs_ps(value);
            let flag = f32::from_bits(_mm_extract_ps::<0>(comparison) as u32);
            assert_eq!(flag.is_nan(), true);
        }
    }

    #[test]
    fn test_nan() {
        unsafe {
            // Test regular
            let value = _mm_set1_ps(23f32);
            let comparison = _mm_isnan_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_eq!(flag, 0);
        }

        unsafe {
            // Test negative regular
            let value = _mm_set1_ps(-23f32);
            let comparison = _mm_isnan_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_eq!(flag, 0);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_ps(f32::NEG_INFINITY);
            let comparison = _mm_isnan_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_eq!(flag, 0);
        }

        unsafe {
            // Test Infinity
            let value = _mm_set1_ps(f32::INFINITY);
            let comparison = _mm_isnan_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_eq!(flag, 0);
        }

        unsafe {
            // Test NaN
            let value = _mm_set1_ps(f32::NAN);
            let comparison = _mm_isnan_ps(value);
            let flag = _mm_extract_ps::<0>(comparison) as u32;
            assert_ne!(flag, 0);
        }
    }
}
