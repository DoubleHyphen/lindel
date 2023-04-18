use criterion::{criterion_group, criterion_main, Criterion};
use lindel::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("hilbert_decode", |b| {
        b.iter(|| {
            let _: [u32; 3] = hilbert_decode(34589430);
        })
    });
    c.bench_function("hilbert_encode", |b| {
        b.iter(|| {
            let _: u128 = hilbert_encode([43u32, 12, 32]);
        })
    });

    c.bench_function("morton_decode", |b| {
        b.iter(|| {
            let _: [u32; 3] = morton_decode(34589430);
        })
    });
    c.bench_function("morton_encode", |b| {
        b.iter(|| {
            let _: u128 = morton_encode([43u32, 12, 32]);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
