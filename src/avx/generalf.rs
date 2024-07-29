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
/// Rounds and takes integral part from float
pub unsafe fn _mm256_rint_ps(f: __m256) -> __m256i {
    let k = _mm256_round_ps::<0x00>(f);
    _mm256_cvtps_epi32(k)
}

#[cfg(not(target_feature = "fma"))]
#[inline]
/// Computes b*c + a using fma when available
pub unsafe fn _mm256_prefer_fma_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    return _mm256_add_ps(_mm256_mul_ps(b, c), a);
}

#[cfg(target_feature = "fma")]
#[inline]
/// Computes b*c + a using fma when available
pub unsafe fn _mm256_prefer_fma_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    return _mm256_fmadd_ps(b, c, a);
}

#[inline(always)]
/// Computes a*b + c
pub unsafe fn _mm256_mlaf_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_prefer_fma_ps(c, b, a)
}

#[inline(always)]
/// Computes 2^n in f32 form for signed 32 bits integers, returns f32 in bits
pub unsafe fn _mm256_pow2if_epi32(n: __m256i) -> __m256i {
    let j = _mm256_slli_epi32::<23>(_mm256_add_epi32(n, _mm256_set1_epi32(0x7f)));
    j
}

#[inline(always)]
/// If mask then `true_vals` otherwise `false_val`
pub unsafe fn _mm256_select_ps(mask: __m256, true_vals: __m256, false_vals: __m256) -> __m256 {
    _mm256_blendv_ps(false_vals, true_vals, mask)
}

#[inline(always)]
/// If mask then `true_vals` otherwise `false_val`
pub unsafe fn _mm256_extract_ps<const IMM8: i32>(x: __m256) -> f32 {
    f32::from_bits(_mm256_extract_epi32::<IMM8>(_mm256_castps_si256(x)) as u32)
}

#[inline(always)]
/// Returns flag value is zero
pub unsafe fn _mm256_eqzero_ps(d: __m256) -> __m256 {
    return _mm256_cmp_ps::<_CMP_EQ_OS>(d, _mm256_set1_ps(0.));
}

#[inline(always)]
/// Founds n in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn _mm256_ilogb2kq_ps(d: __m256) -> __m256i {
    _mm256_sub_epi32(
        _mm256_and_si256(
            _mm256_srli_epi32::<23>(_mm256_castps_si256(d)),
            _mm256_set1_epi32(0xff),
        ),
        _mm256_set1_epi32(0x7f),
    )
}

#[inline(always)]
/// Founds a in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn _mm256_ldexp3kq_ps(x: __m256, n: __m256i) -> __m256 {
    _mm256_castsi256_ps(_mm256_add_epi32(
        _mm256_castps_si256(x),
        _mm256_slli_epi32::<23>(n),
    ))
}

#[inline(always)]
/// Modulus operator for f32
pub unsafe fn _mm256_abs_ps(f: __m256) -> __m256 {
    return _mm256_castsi256_ps(_mm256_andnot_si256(
        _mm256_castps_si256(_mm256_set1_ps(-0.0f32)),
        _mm256_castps_si256(f),
    ));
}

#[inline(always)]
/// Returns flag value is Infinity
pub unsafe fn _mm256_isinf_ps(d: __m256) -> __m256 {
    return _mm256_cmp_ps::<_CMP_EQ_OS>(_mm256_abs_ps(d), _mm256_set1_ps(f32::INFINITY));
}

#[inline(always)]
/// Returns true flag if value is NaN
pub unsafe fn _mm256_isnan_ps(d: __m256) -> __m256 {
    return _mm256_cmp_ps::<_CMP_NEQ_OS>(d, d);
}

#[inline(always)]
/// Returns flag value is lower than zero
pub unsafe fn _mm256_ltzero_ps(d: __m256) -> __m256 {
    return _mm256_cmp_ps::<_CMP_LT_OS>(d, _mm256_set1_ps(0.));
}

#[inline(always)]
/// If mask then `true_vals` otherwise `false_val`
pub unsafe fn _mm256_selecti_ps(mask: __m256i, true_vals: __m256, false_vals: __m256) -> __m256 {
    _mm256_blendv_ps(false_vals, true_vals, _mm256_castsi256_ps(mask))
}

#[inline(always)]
/// Negates value
pub unsafe fn _mm256_neg_ps(f: __m256) -> __m256 {
    _mm256_sub_ps(_mm256_set1_ps(0.), f)
}

#[inline(always)]
/// Copies sign from `y` to `x`
pub unsafe fn _mm256_copysign_ps(x: __m256, y: __m256) -> __m256 {
    _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_andnot_si256(
            _mm256_castps_si256(_mm256_set1_ps(-0.0f32)),
            _mm256_castps_si256(x),
        ),
        _mm256_and_si256(
            _mm256_castps_si256(_mm256_set1_ps(-0.0f32)),
            _mm256_castps_si256(y),
        ),
    ))
}

#[inline(always)]
/// Returns flag value is Neg Infinity
pub unsafe fn _mm256_isneginf_ps(d: __m256) -> __m256 {
    return _mm256_cmp_ps::<_CMP_EQ_OS>(d, _mm256_set1_ps(f32::NEG_INFINITY));
}

#[inline(always)]
/// Checks if arguments is not integral value
pub unsafe fn _mm256_isnotintegral_ps(d: __m256) -> __m256 {
    return _mm256_cmp_ps::<_CMP_NEQ_OS>(d, _mm256_floor_ps(d));
}
