/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use criterion::{criterion_group, criterion_main, Criterion};
use erydanos::{ArcSin, ArcTan, CubeRoot, Power, Sine, Tangent};
use rand::Rng;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    c.bench_function("Sine Erydanos", |b| {
        b.iter(|| {
            let sine_value: f32 = rng.gen_range(0f32..1.0f32);
            _ = sine_value.esin();
            _ = sine_value.esin();
            _ = sine_value.esin();
            _ = sine_value.esin();
            _ = sine_value.esin();
            _ = sine_value.esin();
            _ = sine_value.esin();
        })
    });
    c.bench_function("Sine libm", |b| {
        b.iter(|| {
            let sine_value: f32 = rng.gen_range(0f32..1.0f32);
            libm::sinf(sine_value);
            libm::sinf(sine_value);
            libm::sinf(sine_value);
            libm::sinf(sine_value);
            libm::sinf(sine_value);
            libm::sinf(sine_value);
            libm::sinf(sine_value);
        })
    });
    c.bench_function("Tan Erydanos", |b| {
        b.iter(|| {
            let tan_value: f32 = rng.gen_range(0f32..1.0f32);
            _ = tan_value.etan();
            _ = tan_value.etan();
            _ = tan_value.etan();
            _ = tan_value.etan();
            _ = tan_value.etan();
            _ = tan_value.etan();
            _ = tan_value.etan();
        })
    });
    c.bench_function("Tan libm", |b| {
        b.iter(|| {
            let tan_value: f32 = rng.gen_range(0f32..1.0f32);
            libm::sinf(tan_value);
            libm::sinf(tan_value);
            libm::sinf(tan_value);
            libm::sinf(tan_value);
            libm::sinf(tan_value);
            libm::sinf(tan_value);
            libm::sinf(tan_value);
        })
    });

    c.bench_function("Cbrt Erydanos", |b| {
        b.iter(|| {
            let tan_value: f32 = rng.gen_range(0f32..100f32);
            _ = tan_value.ecbrt();
            _ = tan_value.ecbrt();
            _ = tan_value.ecbrt();
            _ = tan_value.ecbrt();
            _ = tan_value.ecbrt();
            _ = tan_value.ecbrt();
            _ = tan_value.ecbrt();
        })
    });
    c.bench_function("Cbrt libm", |b| {
        b.iter(|| {
            let tan_value: f32 = rng.gen_range(0f32..100f32);
            libm::cbrtf(tan_value);
            libm::cbrtf(tan_value);
            libm::cbrtf(tan_value);
            libm::cbrtf(tan_value);
            libm::cbrtf(tan_value);
            libm::cbrtf(tan_value);
            libm::cbrtf(tan_value);
        })
    });

    c.bench_function("Pow Erydanos", |b| {
        b.iter(|| {
            let tan_value: f32 = rng.gen_range(0f32..100f32);
            _ = tan_value.epow(7.324f32);
            _ = tan_value.epow(7.324f32);
            _ = tan_value.epow(7.324f32);
            _ = tan_value.epow(7.324f32);
            _ = tan_value.epow(7.324f32);
            _ = tan_value.epow(7.324f32);
            _ = tan_value.epow(7.324f32);
        })
    });
    c.bench_function("Pow libm", |b| {
        b.iter(|| {
            let tan_value: f32 = rng.gen_range(0f32..100f32);
            libm::powf(tan_value, 7.324f32);
            libm::powf(tan_value, 7.324f32);
            libm::powf(tan_value, 7.324f32);
            libm::powf(tan_value, 7.324f32);
            libm::powf(tan_value, 7.324f32);
            libm::powf(tan_value, 7.324f32);
            libm::powf(tan_value, 7.324f32);
        })
    });

    c.bench_function("Asin Erydanos", |b| {
        b.iter(|| {
            let tan_value: f32 = rng.gen_range(0f32..1f32);
            _ = tan_value.easin();
            _ = tan_value.easin();
            _ = tan_value.easin();
            _ = tan_value.easin();
            _ = tan_value.easin();
            _ = tan_value.easin();
            _ = tan_value.easin();
        })
    });
    c.bench_function("Asin libm", |b| {
        b.iter(|| {
            let tan_value: f32 = rng.gen_range(0f32..1f32);
            libm::asinf(tan_value);
            libm::asinf(tan_value);
            libm::asinf(tan_value);
            libm::asinf(tan_value);
            libm::asinf(tan_value);
            libm::asinf(tan_value);
            libm::asinf(tan_value);
        })
    });

    c.bench_function("Atan Erydanos", |b| {
        b.iter(|| {
            let tan_value: f32 = rng.gen_range(0f32..100f32);
            _ = tan_value.eatan();
            _ = tan_value.eatan();
            _ = tan_value.eatan();
            _ = tan_value.eatan();
            _ = tan_value.eatan();
            _ = tan_value.eatan();
            _ = tan_value.eatan();
        })
    });
    c.bench_function("Atan libm", |b| {
        b.iter(|| {
            let tan_value: f32 = rng.gen_range(0f32..100f32);
            libm::atanf(tan_value);
            libm::atanf(tan_value);
            libm::atanf(tan_value);
            libm::atanf(tan_value);
            libm::atanf(tan_value);
            libm::atanf(tan_value);
            libm::atanf(tan_value);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
