/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
mod cbrtf;
mod cosf;
mod epi32;
mod epi64;
mod expf;
mod general;
mod hypotf;
mod lnf;
mod powf;
mod saturation_epi32;
mod saturation_epi64;
mod sinf;
mod unsigned;
mod atanf;
mod atan2f;
mod fmod;

pub use cbrtf::_mm_cbrt_ps;
pub use cosf::_mm_cos_ps;
pub use epi32::_mm_blendv_epi32;
pub use epi32::_mm_neg_epi32;
pub use epi32::_mm_select_epi32;
pub use epi64::_mm_abs_epi64;
pub use epi64::_mm_blendv_epi64;
pub use epi64::_mm_mul_epi64;
pub use epi64::_mm_mul_epu64;
pub use epi64::_mm_neg_epi64;
pub use epi64::_mm_select_epi64;
pub use epi64::_mm_setr_epi64x;
pub use expf::_mm_exp_ps;
pub use expf::_mm_expq_fast_ps;
pub use general::_mm_abs_ps;
pub use general::_mm_copysign_ps;
pub use general::_mm_eqzero_ps;
pub use general::_mm_ilogb2kq_ps;
pub use general::_mm_isinf_ps;
pub use general::_mm_isnan_ps;
pub use general::_mm_isneginf_ps;
pub use general::_mm_ldexp3kq_ps;
pub use general::_mm_ltzero_ps;
pub use general::_mm_mlaf_ps;
pub use general::_mm_neg_ps;
pub use general::_mm_packus_epi64;
pub use general::_mm_pow2if_epi32;
pub use general::_mm_prefer_fma_ps;
pub use general::_mm_rint_ps;
pub use general::_mm_select_ps;
pub use general::_mm_selecti_ps;
pub use hypotf::_mm_hypot_fast_ps;
pub use hypotf::_mm_hypot_ps;
pub use lnf::_mm_ln_fast_ps;
pub use lnf::_mm_ln_ps;
pub use powf::_mm_pow_fast_ps;
pub use powf::_mm_pow_ps;
pub use atanf::_mm_atan_ps;
pub use atan2f::_mm_atan2_ps;
pub use fmod::_mm_fmod_ps;
pub use saturation_epi32::_mm_adds_epi32;
pub use saturation_epi32::_mm_adds_epu32;
pub use saturation_epi32::_mm_subs_epi32;
pub use saturation_epi64::_mm_adds_epi64;
pub use saturation_epi64::_mm_adds_epu64;
pub use saturation_epi64::_mm_subs_epi64;
pub use sinf::_mm_sin_ps;
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
