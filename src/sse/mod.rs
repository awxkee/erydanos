/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
mod acos;
mod acosf;
mod asin;
mod asinf;
mod atan;
mod atan2;
mod atan2f;
mod atanf;
mod cbrtf;
mod cos;
mod cosf;
mod epi32;
mod epi64;
mod exp;
mod expf;
mod fmod;
mod fmodf;
mod general;
mod generalf;
mod hypot;
mod hypot3;
mod hypot3f;
mod hypot4f;
mod hypotf;
mod ln;
mod lnf;
mod pow;
mod powf;
mod saturation_epi32;
mod saturation_epi64;
mod sin;
mod sinf;
mod tan;
mod tanf;
mod unsigned;

pub use acos::_mm_acos_pd;
pub use acosf::_mm_acos_ps;
pub use asin::_mm_asin_pd;
pub use asinf::_mm_asin_ps;
pub use atan::_mm_atan_pd;
pub use atan2::_mm_atan2_pd;
pub use atan2f::_mm_atan2_ps;
pub use atanf::_mm_atan_ps;
pub use cbrtf::_mm_cbrt_ps;
pub use cos::_mm_cos_pd;
pub use cosf::_mm_cos_ps;
pub use epi32::_mm_blendv_epi32;
pub use epi32::_mm_neg_epi32;
pub use epi32::_mm_select_epi32;
pub use epi64::_mm_abs_epi64;
pub use epi64::_mm_blendv_epi64;
pub use epi64::_mm_cvtepi64_pd;
pub use epi64::_mm_mul_epi64;
pub use epi64::_mm_mul_epu64;
pub use epi64::_mm_neg_epi64;
pub use epi64::_mm_select_epi64;
pub use epi64::_mm_setr_epi64x;
pub use epi64::_mm_sllv_epi64x;
pub use epi64::_mm_srlv_epi64x;
pub use exp::_mm_exp_pd;
pub use exp::_mm_expq_fast_pd;
pub use expf::_mm_exp_ps;
pub use expf::_mm_expq_fast_ps;
pub use fmod::_mm_fmod_pd;
pub use fmodf::_mm_fmod_ps;
pub use general::_mm_abs_pd;
pub use general::_mm_copysign_pd;
pub use general::_mm_cvtpd_epi64;
pub use general::_mm_cvtpd_epu64;
pub use general::_mm_eqzero_pd;
pub use general::_mm_extract_pd;
pub use general::_mm_isinf_pd;
pub use general::_mm_isintegral_pd;
pub use general::_mm_isnan_pd;
pub use general::_mm_isneginf_pd;
pub use general::_mm_isnotintegral_pd;
pub use general::_mm_mlaf_pd;
pub use general::_mm_neg_pd;
pub use general::_mm_prefer_fma_pd;
pub use general::_mm_select_pd;
pub use general::_mm_selecti_pd;
pub use generalf::_mm_abs_ps;
pub use generalf::_mm_copysign_ps;
pub use generalf::_mm_eqzero_ps;
pub use generalf::_mm_ilogb2kq_ps;
pub use generalf::_mm_isinf_ps;
pub use generalf::_mm_isintegral_ps;
pub use generalf::_mm_isnan_ps;
pub use generalf::_mm_isneginf_ps;
pub use generalf::_mm_isnotintegral_ps;
pub use generalf::_mm_ldexp3kq_ps;
pub use generalf::_mm_ltzero_ps;
pub use generalf::_mm_mlaf_ps;
pub use generalf::_mm_neg_ps;
pub use generalf::_mm_packus_epi64;
pub use generalf::_mm_pow2if_epi32;
pub use generalf::_mm_prefer_fma_ps;
pub use generalf::_mm_rint_ps;
pub use generalf::_mm_select_ps;
pub use generalf::_mm_selecti_ps;
pub use hypot::_mm_hypot_fast_pd;
pub use hypot::_mm_hypot_pd;
pub use hypot3::_mm_hypot3_fast_pd;
pub use hypot3::_mm_hypot3_pd;
pub use hypot3f::_mm_hypot3_fast_ps;
pub use hypot3f::_mm_hypot3_ps;
pub use hypot4f::_mm_hypot4_fast_ps;
pub use hypot4f::_mm_hypot4_ps;
pub use hypotf::_mm_hypot_fast_ps;
pub use hypotf::_mm_hypot_ps;
pub use ln::_mm_ln_fast_pd;
pub use ln::_mm_ln_pd;
pub use lnf::_mm_ln_fast_ps;
pub use lnf::_mm_ln_ps;
pub use pow::_mm_pow_fast_pd;
pub use pow::_mm_pow_pd;
pub use powf::_mm_pow_fast_ps;
pub use powf::_mm_pow_ps;
pub use saturation_epi32::_mm_adds_epi32;
pub use saturation_epi32::_mm_adds_epu32;
pub use saturation_epi32::_mm_subs_epi32;
pub use saturation_epi64::_mm_adds_epi64;
pub use saturation_epi64::_mm_adds_epu64;
pub use saturation_epi64::_mm_subs_epi64;
pub use sin::_mm_sin_pd;
pub use sinf::_mm_sin_ps;
pub use tan::_mm_tan_pd;
pub use tanf::_mm_tan_ps;
pub use unsigned::_mm_cmpge_epi64;
pub use unsigned::_mm_cmpge_epu16;
pub use unsigned::_mm_cmpge_epu32;
pub use unsigned::_mm_cmpge_epu64;
pub use unsigned::_mm_cmpge_epu8;
pub use unsigned::_mm_cmpgt_epu16;
pub use unsigned::_mm_cmpgt_epu32;
pub use unsigned::_mm_cmpgt_epu64;
pub use unsigned::_mm_cmpgt_epu8;
pub use unsigned::_mm_cmple_epi64;
pub use unsigned::_mm_cmple_epu16;
pub use unsigned::_mm_cmple_epu32;
pub use unsigned::_mm_cmple_epu64;
pub use unsigned::_mm_cmple_epu8;
pub use unsigned::_mm_cmplt_epi64;
pub use unsigned::_mm_cmplt_epu16;
pub use unsigned::_mm_cmplt_epu32;
pub use unsigned::_mm_cmplt_epu64;
pub use unsigned::_mm_cmplt_epu8;
pub use unsigned::_mm_max_epi64x;
pub use unsigned::_mm_min_epi64x;

pub const fn _mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}
