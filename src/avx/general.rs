use crate::shuffle::_mm_shuffle;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(not(target_feature = "fma"))]
#[inline]
/// Computes b*c + a using fma when available
pub unsafe fn _mm256_prefer_fma_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    return _mm256_add_pd(_mm_mul_pd(b, c), a);
}

#[cfg(target_feature = "fma")]
#[inline]
/// Computes b*c + a using fma when available
pub unsafe fn _mm256_prefer_fma_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    return _mm256_fmadd_pd(b, c, a);
}

#[inline(always)]
/// Computes a*b + c
pub unsafe fn _mm256_mlaf_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    _mm256_prefer_fma_pd(c, b, a)
}

#[inline(always)]
/// Modulus operator for f64
pub unsafe fn _mm256_abs_pd(f: __m256d) -> __m256d {
    return _mm256_castsi256_pd(_mm256_andnot_si256(
        _mm256_castpd_si256(_mm256_set1_pd(-0.0f64)),
        _mm256_castpd_si256(f),
    ));
}

#[inline(always)]
/// Negates value
pub unsafe fn _mm256_neg_pd(f: __m256d) -> __m256d {
    _mm256_sub_pd(_mm256_set1_pd(0.), f)
}

#[inline(always)]
/// Rounds and takes integral part 64 bytes from double
pub unsafe fn _mm256_rint_pd(f: __m256d) -> __m256i {
    let k = _mm256_round_pd::<0x00>(f);
    _mm256_cvtpd_epi64x(k)
}

#[inline(always)]
// Converts double to integers 64
pub unsafe fn _mm256_cvtpd_epu64x(v: __m256d) -> __m256i {
    let k_k513ff = _mm256_set1_epi64x(51 + 0x3FF);

    // Exponent indicates whether the number can be represented as uuint64_t.
    let biased_exp = _mm256_and_si256(
        _mm256_srli_epi64::<52>(_mm256_castpd_si256(v)),
        _mm256_set1_epi64x(0x7FF),
    );
    let mantissa = _mm256_and_si256(_mm256_castpd_si256(v), _mm256_set1_epi64x((1 << 52) - 1));

    // Calculate left and right shifts to move mantissa into place.
    let shift_right = _mm256_subs_epu16(k_k513ff, biased_exp);
    let shift_left = _mm256_subs_epu16(biased_exp, k_k513ff);

    // Shift mantissa into place.
    let shifted = _mm256_srli_epi64::<1>(_mm256_srlv_epi64(
        _mm256_sllv_epi64(mantissa, shift_left),
        shift_right,
    ));
    // Include implicit 1-bit.
    let implicit_bit_shifted = _mm256_srlv_epi64(
        _mm256_sllv_epi64(_mm256_set1_epi64x(1 << 51), shift_left),
        shift_right,
    );
    let magnitude = _mm256_or_si256(shifted, implicit_bit_shifted);

    const SH_1: i32 = _mm_shuffle(3, 3, 1, 1);
    // Fill each 64-bit part with sign bits.
    let sign_mask = _mm256_shuffle_epi32::<SH_1>(_mm256_srai_epi32::<31>(_mm256_castpd_si256(v)));
    // Mask out negative values to 0.
    let lower_bounded = _mm256_andnot_si256(sign_mask, magnitude);

    const SH_2: i32 = _mm_shuffle(2, 2, 0, 0);
    // 0xFF is exp < 64
    let upper_bound_mask = _mm256_shuffle_epi32::<SH_2>(_mm256_cmpgt_epi32(
        _mm256_set1_epi32(64 + 0x3FF),
        biased_exp,
    ));
    // Mask out overflow values to 0.
    let fully_bounded = _mm256_and_si256(lower_bounded, upper_bound_mask);

    return fully_bounded;
}

#[inline(always)]
// Converts double to integers 64
pub unsafe fn _mm256_cvtpd_epi64x(v: __m256d) -> __m256i {
    let k_513ff = _mm256_set1_epi64x(51 + 0x3FF);

    // Exponent indicates whether the number can be represented as uuint64_t.
    let biased_exp = _mm256_and_si256(
        _mm256_srli_epi64::<52>(_mm256_castpd_si256(v)),
        _mm256_set1_epi64x(0x7FF),
    );
    let mantissa = _mm256_and_si256(_mm256_castpd_si256(v), _mm256_set1_epi64x((1 << 52) - 1));

    // Calculate left and right shifts to move mantissa into place.
    let shift_right = _mm256_subs_epu16(k_513ff, biased_exp);
    let shift_left = _mm256_subs_epu16(biased_exp, k_513ff);

    // Shift mantissa into place.
    let shifted = _mm256_srli_epi64::<1>(_mm256_srlv_epi64(
        _mm256_sllv_epi64(mantissa, shift_left),
        shift_right,
    ));
    // Include implicit 1-bit.
    let implicit_bit_shifted = _mm256_srlv_epi64(
        _mm256_sllv_epi64(_mm256_set1_epi64x(1 << 51), shift_left),
        shift_right,
    );
    let magnitude = _mm256_or_si256(shifted, implicit_bit_shifted);

    const SH_1: i32 = _mm_shuffle(3, 3, 1, 1);

    // Fill each 64-bit part with sign bits.
    let sign_mask = _mm256_shuffle_epi32::<SH_1>(_mm256_srai_epi32::<31>(_mm256_castpd_si256(v)));
    // Adjust for negative values.
    let sign_adjusted = _mm256_sub_epi64(_mm256_xor_si256(magnitude, sign_mask), sign_mask);

    const SH_2: i32 = _mm_shuffle(2, 2, 0, 0);

    // 0xFF is exp < 64
    let upper_bound_mask = _mm256_shuffle_epi32::<SH_2>(_mm256_cmpgt_epi32(
        _mm256_set1_epi32(64 + 0x3FF),
        biased_exp,
    ));
    // Saturate overflow values to INT64_MIN.
    let bounded = _mm256_blendv_epi8(
        _mm256_set1_epi64x(i64::MAX),
        sign_adjusted,
        upper_bound_mask,
    );

    return bounded;
}

#[inline(always)]
/// Returns true flag if value is NaN
pub unsafe fn _mm256_isnan_pd(d: __m256d) -> __m256d {
    return _mm256_cmp_pd::<_CMP_NEQ_OS>(d, d);
}

#[inline(always)]
/// Returns flag value is Infinity
pub unsafe fn _mm256_isinf_pd(d: __m256d) -> __m256d {
    return _mm256_cmp_pd::<_CMP_EQ_OS>(_mm256_abs_pd(d), _mm256_set1_pd(f64::INFINITY));
}

#[inline(always)]
// Converts integers 64 to double
pub unsafe fn _mm256_cvtepi64_pdx(v: __m256i) -> __m256d {
    let magic_i_lo = _mm256_set1_epi64x(0x4330000000000000); // 2^52               encoded as floating-point
    let magic_i_hi32 = _mm256_set1_epi64x(0x4530000080000000); // 2^84 + 2^63        encoded as floating-point
    let magic_i_all = _mm256_set1_epi64x(0x4530000080100000); // 2^84 + 2^63 + 2^52 encoded as floating-point
    let magic_d_all = _mm256_castsi256_pd(magic_i_all);

    let v_lo = _mm256_blend_epi32::<0b01010101>(magic_i_lo, v); // Blend the 32 lowest significant bits of v with magic_int_lo
    let mut v_hi = _mm256_srli_epi64::<32>(v); // Extract the 32 most significant bits of v
    v_hi = _mm256_xor_si256(v_hi, magic_i_hi32); // Flip the msb of v_hi and blend with 0x45300000
    let v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); // Compute in double precision:
    _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo)) // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

#[inline(always)]
/// If mask then `true_vals` otherwise `false_val`
pub unsafe fn _mm256_select_pd(mask: __m256d, true_vals: __m256d, false_vals: __m256d) -> __m256d {
    _mm256_blendv_pd(false_vals, true_vals, mask)
}

#[inline(always)]
/// If mask then `true_vals` otherwise `false_val`
pub unsafe fn _mm256_selecti_pd(mask: __m256i, true_vals: __m256d, false_vals: __m256d) -> __m256d {
    _mm256_blendv_pd(false_vals, true_vals, _mm256_castsi256_pd(mask))
}

#[inline(always)]
/// Extracts f64 value
pub unsafe fn _mm256_extract_pd<const IMM: i32>(d: __m256d) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        f64::from_bits(_mm256_extract_epi64::<IMM>(_mm256_castpd_si256(d)) as u64)
    }
    #[cfg(target_arch = "x86")]
    {
        let (low, high);
        let rc = _mm256_castpd_si256(d);
        if IMM == 0 {
            low = _mm256_cvtsi256_si32(rc);
            high = _mm256_cvtsi256_si32(_mm256_srli_si256::<4>(rc));
        } else if IMM == 1 {
            low = _mm_cvtsi128_si32(_mm256_srli_si256::<8>(rc));
            high = _mm_cvtsi128_si32(_mm256_srli_si256::<12>(rc));
        } else if IMM == 2 {
            low = _mm_cvtsi128_si32(_mm256_srli_si256::<16>(rc));
            high = _mm_cvtsi128_si32(_mm256_srli_si256::<20>(rc));
        } else {
            low = _mm_cvtsi128_si32(_mm256_srli_si256::<24>(rc));
            high = _mm_cvtsi128_si32(_mm256_srli_si256::<28>(rc));
        }
        return f64::from_bits(((high as u64) << 32) | low as u64);
    }
}

#[inline(always)]
/// Founds n in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn _mm256_ilogb2k_pd(d: __m256d) -> __m256i {
    _mm256_sub_epi64(
        _mm256_and_si256(
            _mm256_srli_epi64::<52>(_mm256_castpd_si256(d)),
            _mm256_set1_epi64x(0x7ff),
        ),
        _mm256_set1_epi64x(0x3ff),
    )
}

#[inline(always)]
/// Founds a in x=a+𝑛ln(2), |a| <= 1
pub unsafe fn _mm256_ldexp3k_pd(x: __m256d, n: __m256i) -> __m256d {
    _mm256_castsi256_pd(_mm256_add_epi64(
        _mm256_castpd_si256(x),
        _mm256_slli_epi64::<52>(n),
    ))
}

#[inline(always)]
/// Computes 2^n in f64 form for signed 64 bits integers, returns f64 in bits
pub unsafe fn _mm256_pow2i_epi64(n: __m256i) -> __m256i {
    let j = _mm256_slli_epi64::<52>(_mm256_add_epi64(n, _mm256_set1_epi32(0x3ff)));
    j
}

#[inline(always)]
/// Copies sign from `y` to `x`
pub unsafe fn _mm256_copysign_pd(x: __m256d, y: __m256d) -> __m256d {
    _mm256_castsi256_pd(_mm256_xor_si256(
        _mm256_andnot_si256(
            _mm256_castpd_si256(_mm256_set1_pd(-0.0f64)),
            _mm256_castpd_si256(x),
        ),
        _mm256_and_si256(
            _mm256_castpd_si256(_mm256_set1_pd(-0.0f64)),
            _mm256_castpd_si256(y),
        ),
    ))
}

#[inline(always)]
/// Returns flag value is Neg Infinity
pub unsafe fn _mm256_isneginf_pd(d: __m256d) -> __m256d {
    return _mm256_cmp_pd::<_CMP_EQ_OS>(d, _mm256_set1_pd(f64::NEG_INFINITY));
}

#[inline(always)]
/// Checks if arguments is not integral value
pub unsafe fn _mm256_isnotintegral_pd(d: __m256d) -> __m256d {
    return _mm256_cmp_pd::<_CMP_NEQ_OS>(d, _mm256_floor_pd(d));
}