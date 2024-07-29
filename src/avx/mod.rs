/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

mod acosf;
mod asinf;
mod atan2f;
mod atanf;
mod cbrtf;
mod cosf;
mod epi32;
mod epi64;
mod expf;
mod fmodf;
mod generalf;
mod hypot3f;
mod hypot4f;
mod hypotf;
mod lnf;
mod powf;
mod sinf;
mod tanf;

pub use acosf::_mm256_acos_ps;
pub use asinf::_mm256_asin_ps;
pub use atan2f::_mm256_atan2_ps;
pub use atanf::_mm256_atan_ps;
pub use cbrtf::_mm256_cbrt_fast_ps;
pub use cbrtf::_mm256_cbrt_ps;
pub use cosf::_mm256_cos_ps;
pub use epi32::_mm256_neg_epi32;
pub use epi64::_mm256_cvtepi64_epi32x;
pub use epi64::_mm256_max_epi64x;
pub use epi64::_mm256_min_epi64x;
pub use epi64::_mm256_mul_epi64;
pub use epi64::_mm256_mul_epu64;
pub use epi64::_mm256_packts_epi64;
pub use epi64::_mm256_packus_epi64;
pub use epi64::_mm256_srai_epi64x;
pub use expf::_mm256_exp_ps;
pub use expf::_mm256_expq_fast_ps;
pub use fmodf::_mm256_fmod_ps;
pub use generalf::_mm256_abs_ps;
pub use generalf::_mm256_eqzero_ps;
pub use generalf::_mm256_extract_ps;
pub use generalf::_mm256_ilogb2kq_ps;
pub use generalf::_mm256_isinf_ps;
pub use generalf::_mm256_isnan_ps;
pub use generalf::_mm256_isneginf_ps;
pub use generalf::_mm256_isnotintegral_ps;
pub use generalf::_mm256_ldexp3kq_ps;
pub use generalf::_mm256_ltzero_ps;
pub use generalf::_mm256_mlaf_ps;
pub use generalf::_mm256_neg_ps;
pub use generalf::_mm256_pow2if_epi32;
pub use generalf::_mm256_prefer_fma_ps;
pub use generalf::_mm256_rint_ps;
pub use generalf::_mm256_select_ps;
pub use generalf::_mm256_selecti_ps;
pub use hypot3f::_mm256_hypot3_fast_ps;
pub use hypot3f::_mm256_hypot3_ps;
pub use hypot4f::_mm256_hypot4_fast_ps;
pub use hypot4f::_mm256_hypot4_ps;
pub use hypotf::_mm256_hypot_fast_ps;
pub use hypotf::_mm256_hypot_ps;
pub use lnf::_mm256_ln_fast_ps;
pub use lnf::_mm256_ln_ps;
pub use powf::_mm256_pow_fast_ps;
pub use powf::_mm256_pow_ps;
pub use sinf::_mm256_sin_ps;
pub use tanf::_mm256_tan_ps;
