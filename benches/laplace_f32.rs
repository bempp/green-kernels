use criterion::{criterion_group, criterion_main, Criterion};
use itertools;
use pulp::f32x4;

extern crate blas_src;
extern crate lapack_src;

use core::arch::aarch64;
use std::arch::aarch64::float32x4x3_t;

use num::traits::FloatConst;

use rlst::prelude::*;

use green_kernels::laplace_3d::Laplace3dKernel;
use green_kernels::traits::Kernel;
use green_kernels::types::EvalType;

use rand::SeedableRng;

const NPOINTS: usize = 1000;

pub fn laplace_f32_test_standard(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

    let mut sources = rlst_dynamic_array2!(f32, [3, NPOINTS]);
    let mut targets = rlst_dynamic_array2!(f32, [3, NPOINTS]);

    let mut charges = rlst_dynamic_array1!(f32, [NPOINTS]);

    let mut result = rlst_dynamic_array1!(f32, [NPOINTS]);

    sources.fill_from_equally_distributed(&mut rng);
    targets.fill_from(sources.view());

    charges.fill_from_standard_normal(&mut rng);

    c.bench_function("f32 auto vectorized", |b| {
        b.iter(|| {
            Laplace3dKernel::<f32>::new().evaluate_st(
                EvalType::Value,
                sources.data(),
                targets.data(),
                charges.data(),
                result.data_mut(),
            );
        })
    });
}

pub fn laplace_f32_test_simd(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

    let mut sources = rlst_dynamic_array2!(f32, [3, NPOINTS]);
    let mut targets = rlst_dynamic_array2!(f32, [3, NPOINTS]);

    let mut charges = rlst_dynamic_array1!(f32, [NPOINTS]);

    let mut result = rlst_dynamic_array1!(f32, [NPOINTS]);

    sources.fill_from_equally_distributed(&mut rng);
    targets.fill_from(sources.view());

    charges.fill_from_standard_normal(&mut rng);

    c.bench_function("f32 manual simd", |b| {
        b.iter(|| {
            evaluate_laplace_st_simd(
                EvalType::Value,
                sources.data(),
                targets.data(),
                charges.data(),
                result.data_mut(),
            );
        })
    });
}

pub fn laplace_f32_test_row_major_simd(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

    let mut sources = rlst_dynamic_array2!(f32, [3, NPOINTS]);
    let mut targets = rlst_dynamic_array2!(f32, [3, NPOINTS]);

    let mut charges = rlst_dynamic_array1!(f32, [NPOINTS]);

    let mut result = rlst_dynamic_array1!(f32, [NPOINTS]);

    sources.fill_from_equally_distributed(&mut rng);
    targets.fill_from(sources.view());

    charges.fill_from_standard_normal(&mut rng);

    c.bench_function("f32 manual row major simd", |b| {
        b.iter(|| {
            evaluate_laplace_st_row_major_simd(
                EvalType::Value,
                sources.data(),
                targets.data(),
                charges.data(),
                result.data_mut(),
            );
        })
    });
}

pub fn laplace_f32_test_by_col(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

    let mut sources = rlst_dynamic_array2!(f32, [3, NPOINTS]);
    let mut targets = rlst_dynamic_array2!(f32, [3, NPOINTS]);

    let mut charges = rlst_dynamic_array1!(f32, [NPOINTS]);

    let mut result = rlst_dynamic_array1!(f32, [NPOINTS]);

    sources.fill_from_equally_distributed(&mut rng);
    targets.fill_from(sources.view());

    charges.fill_from_standard_normal(&mut rng);

    c.bench_function("f32 by col", |b| {
        b.iter(|| {
            evaluate_laplace_by_col(
                sources.data(),
                targets.data(),
                charges.data(),
                result.data_mut(),
            );
        })
    });
}

fn evaluate_laplace_by_col(sources: &[f32], targets: &[f32], charges: &[f32], result: &mut [f32]) {
    let sources = rlst_array_from_slice2!(sources, [3, NPOINTS]);
    let targets = rlst_array_from_slice2!(targets, [3, NPOINTS]);
    let charges = rlst_array_from_slice1!(charges, [NPOINTS]);

    let laplace = Laplace3dKernel::<f32>::new();
    let mut res: [f32; 1] = [0.0];

    for (target_index, target) in targets.view().col_iter().enumerate() {
        for (source_index, source) in sources.view().col_iter().enumerate() {
            laplace.greens_fct(EvalType::Value, source.data(), target.data(), &mut res);
            result[target_index] += charges[[source_index]] * res[0];
        }
    }
}

fn evaluate_laplace_st_row_major_simd(
    eval_type: EvalType,
    sources: &[f32],
    targets: &[f32],
    charges: &[f32],
    result: &mut [f32],
) {
    for index in 0..NPOINTS {
        let target = [
            targets[index],
            targets[NPOINTS + index],
            targets[2 * NPOINTS + index],
        ];

        let mut res: [f32; 1] = [0.0];

        evaluate_laplace_one_target_row_major_simd_new(
            eval_type, &target, sources, charges, &mut res,
        );
        result[index] = res[0];
    }
}

fn evaluate_laplace_st_simd(
    eval_type: EvalType,
    sources: &[f32],
    targets: &[f32],
    charges: &[f32],
    result: &mut [f32],
) {
    for index in 0..NPOINTS {
        let target = [
            targets[index],
            targets[NPOINTS + index],
            targets[2 * NPOINTS + index],
        ];

        let mut res: [f32; 1] = [0.0];

        evaluate_laplace_one_target_simd(eval_type, &target, sources, charges, &mut res);
        result[index] = res[0];
    }
}
///
/// Evaluate laplce kernel with one target
pub fn evaluate_laplace_one_target_simd(
    eval_type: EvalType,
    target: &[f32],
    sources: &[f32],
    charges: &[f32],
    result: &mut [f32],
) {
    let ncharges = charges.len();
    let nsources = ncharges;
    let m_inv_4pi = 0.25 * f32::FRAC_1_PI();

    let sources0 = &sources[0..nsources];
    let sources1 = &sources[nsources..2 * nsources];
    let sources2 = &sources[2 * nsources..3 * nsources];

    match eval_type {
        EvalType::Value => {
            for chunk_start in (0..nsources).step_by(4) {
                unsafe {
                    let zero = aarch64::vdupq_n_f32(0.0);
                    let source_x = aarch64::vld1q_dup_f32(&sources0[chunk_start]);
                    let source_y = aarch64::vld1q_dup_f32(&sources1[chunk_start]);
                    let source_z = aarch64::vld1q_dup_f32(&sources2[chunk_start]);

                    let charge = aarch64::vld1q_dup_f32(&charges[chunk_start]);

                    let target_x: aarch64::float32x4_t = aarch64::vdupq_n_f32(target[0]);
                    let target_y: aarch64::float32x4_t = aarch64::vdupq_n_f32(target[1]);
                    let target_z: aarch64::float32x4_t = aarch64::vdupq_n_f32(target[2]);

                    let diff_x = aarch64::vsubq_f32(target_x, source_x);
                    let diff_y = aarch64::vsubq_f32(target_y, source_y);
                    let diff_z = aarch64::vsubq_f32(target_z, source_z);

                    let diff_sq_x = aarch64::vmulq_f32(diff_x, diff_x);
                    let diff_sq_y = aarch64::vmulq_f32(diff_y, diff_y);
                    let diff_sq_z = aarch64::vmulq_f32(diff_z, diff_z);

                    let sum =
                        aarch64::vaddq_f32(aarch64::vaddq_f32(diff_sq_x, diff_sq_y), diff_sq_z);

                    let greater_zero = aarch64::vcgtq_f32(sum, zero);

                    let filtered_inv_sqrt =
                        aarch64::vbslq_f32(greater_zero, aarch64::vrsqrteq_f32(sum), zero);

                    let prod = aarch64::vmulq_n_f32(filtered_inv_sqrt, m_inv_4pi);

                    let prod_with_charge = aarch64::vmulq_f32(prod, charge);
                    result[0] += aarch64::vaddvq_f32(prod_with_charge);
                }
            }
        }
        EvalType::ValueDeriv => {
            panic!("Not supported");
        }
    }
}

/// Evaluate laplce kernel with one target
pub fn evaluate_laplace_one_target_row_major_simd(
    eval_type: EvalType,
    target: &[f32],
    sources: &[f32],
    charges: &[f32],
    result: &mut [f32],
) {
    let ncharges = charges.len();
    let nsources = ncharges;
    let m_inv_4pi = 0.25 * f32::FRAC_1_PI();

    // let sources0 = &sources[0..nsources];
    // let sources1 = &sources[nsources..2 * nsources];
    // let sources2 = &sources[2 * nsources..3 * nsources];

    match eval_type {
        EvalType::Value => {
            for chunk_start in (0..nsources).step_by(4) {
                unsafe {
                    let zero = aarch64::vdupq_n_f32(0.0);
                    let aarch64::float32x4x3_t(source_x, source_y, source_z) =
                        aarch64::vld3q_f32(&sources[chunk_start]);

                    let charge = aarch64::vld1q_dup_f32(&charges[chunk_start]);

                    let target_x: aarch64::float32x4_t = aarch64::vdupq_n_f32(target[0]);
                    let target_y: aarch64::float32x4_t = aarch64::vdupq_n_f32(target[1]);
                    let target_z: aarch64::float32x4_t = aarch64::vdupq_n_f32(target[2]);

                    let diff_x = aarch64::vsubq_f32(target_x, source_x);
                    let diff_y = aarch64::vsubq_f32(target_y, source_y);
                    let diff_z = aarch64::vsubq_f32(target_z, source_z);

                    let diff_sq_x = aarch64::vmulq_f32(diff_x, diff_x);
                    let diff_sq_y = aarch64::vmulq_f32(diff_y, diff_y);
                    let diff_sq_z = aarch64::vmulq_f32(diff_z, diff_z);

                    let sum =
                        aarch64::vaddq_f32(aarch64::vaddq_f32(diff_sq_x, diff_sq_y), diff_sq_z);

                    let greater_zero = aarch64::vcgtq_f32(sum, zero);

                    let filtered_inv_sqrt =
                        aarch64::vbslq_f32(greater_zero, aarch64::vrsqrteq_f32(sum), zero);

                    let prod = aarch64::vmulq_n_f32(filtered_inv_sqrt, m_inv_4pi);

                    let prod_with_charge = aarch64::vmulq_f32(prod, charge);
                    result[0] += aarch64::vaddvq_f32(prod_with_charge);
                }
            }
        }
        EvalType::ValueDeriv => {
            panic!("Not supported");
        }
    }
}

pub fn evaluate_laplace_one_target_row_major_simd_new(
    eval_type: EvalType,
    target: &[f32],
    sources: &[f32],
    charges: &[f32],
    result: &mut [f32],
) {
    let ncharges = charges.len();
    let nsources = ncharges;
    let m_inv_4pi = 0.25 * f32::FRAC_1_PI();

    // let sources0 = &sources[0..nsources];
    // let sources1 = &sources[nsources..2 * nsources];
    // let sources2 = &sources[2 * nsources..3 * nsources];

    struct Impl<'a> {
        simd: pulp::aarch64::Neon,
        t0: f32,
        t1: f32,
        t2: f32,
        sources: &'a [f32],
        charges: &'a [f32],
    }

    impl<'a> pulp::NullaryFnOnce for Impl<'a> {
        type Output = f32;

        fn call(self) -> Self::Output {
            let Self {
                simd,
                t0,
                t1,
                t2,
                sources,
                charges,
            } = self;
            let m_inv_4pi = 0.25 * f32::FRAC_1_PI();

            let (s_head, s_tail) = pulp::as_arrays::<12, _>(sources);
            let (c_head, c_tail) = pulp::as_arrays::<4, _>(charges);

            let mut acc;

            use pulp::Simd;

            {
                let t0v = simd.splat_f32x4(t0);
                let t1v = simd.splat_f32x4(t1);
                let t2v = simd.splat_f32x4(t2);

                let zero = simd.splat_f32x4(0.0);
                let mut res = simd.splat_f32x4(0.0);
                for (s, c) in itertools::izip!(s_head, c_head) {
                    let float32x4x3_t(sx, sy, sz) = unsafe { simd.neon.vld3q_f32(s.as_ptr()) };
                    let sx: f32x4 = unsafe { std::mem::transmute(sx) };
                    let sy: f32x4 = unsafe { std::mem::transmute(sy) };
                    let sz: f32x4 = unsafe { std::mem::transmute(sz) };

                    let c: f32x4 = pulp::cast(*c);

                    let diffx = simd.sub_f32x4(t0v, sx);
                    let diffy = simd.sub_f32x4(t1v, sy);
                    let diffz = simd.sub_f32x4(t2v, sz);

                    let square_sum = simd.mul_add_f32x4(
                        diffz,
                        diffz,
                        simd.mul_add_f32x4(diffy, diffy, simd.mul_f32x4(diffx, diffx)),
                    );

                    let is_zero = simd.cmp_eq_f32x4(square_sum, zero);

                    let approx_inv_sqrt: f32x4 = simd.select_f32x4(is_zero, zero, unsafe {
                        std::mem::transmute(simd.neon.vrsqrteq_f64(std::mem::transmute(square_sum)))
                    });

                    res = simd.mul_add_f32x4(approx_inv_sqrt, c, res);
                }

                acc = simd.f32s_reduce_sum(res);

                let (s_tail, _) = pulp::as_arrays::<3, _>(s_tail);

                for (s, c) in itertools::izip!(s_tail, c_tail) {
                    let diffx = s[0] - t0;
                    let diffy = s[1] - t1;
                    let diffz = s[2] - t2;

                    let square_sum = fma(diffz, diffz, fma(diffy, diffy, diffx * diffx));

                    let inv_root = {
                        if square_sum == 0.0 {
                            0.0
                        } else {
                            1.0 / square_sum.sqrt()
                        }
                    };
                    acc = fma(inv_root, *c, acc);
                }
            }
            acc * m_inv_4pi
        }
    }

    let simd = pulp::aarch64::Neon::try_new().unwrap();

    let acc = simd.vectorize(Impl {
        simd,
        t0: target[0],
        t1: target[1],
        t2: target[2],
        sources,
        charges,
    });

    result[0] = acc;
}

#[inline]
fn fma<T: 'static>(x: T, y: T, z: T) -> T {
    use coe::coerce_static as to;
    if coe::is_same::<T, f32>() {
        to(f32::mul_add(to(x), to(y), to(z)))
    } else if coe::is_same::<T, f64>() {
        to(f64::mul_add(to(x), to(y), to(z)))
    } else {
        panic!()
    }
}

criterion_group!(
    benches,
    laplace_f32_test_simd,
    laplace_f32_test_row_major_simd,
    laplace_f32_test_standard,
    laplace_f32_test_by_col
);
criterion_main!(benches);
