use criterion::{criterion_group, criterion_main, Criterion};

extern crate blas_src;
extern crate lapack_src;

use green_kernels::laplace_3d::Laplace3dKernel;
use green_kernels::traits::Kernel;
use green_kernels::types::GreenKernelEvalType;

use rand::SeedableRng;
use rlst::rlst_dynamic_array;

const NPOINTS: usize = 1000;

pub fn laplace_f32_test_standard(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

    let mut sources = rlst_dynamic_array!(f32, [3, NPOINTS]);
    let mut targets = rlst_dynamic_array!(f32, [3, NPOINTS]);

    let mut charges = rlst_dynamic_array!(f32, [NPOINTS]);

    let mut result = rlst_dynamic_array!(f32, [NPOINTS]);

    sources.fill_from_equally_distributed(&mut rng);
    targets.fill_from(&sources);

    charges.fill_from_standard_normal(&mut rng);

    c.bench_function("Laplace evaluate f32", |b| {
        b.iter(|| {
            Laplace3dKernel::<f32>::new().evaluate_st(
                GreenKernelEvalType::Value,
                sources.data().unwrap(),
                targets.data().unwrap(),
                charges.data().unwrap(),
                result.data_mut().unwrap(),
            );
        })
    });
}

criterion_group!(benches, laplace_f32_test_standard,);
criterion_main!(benches);
