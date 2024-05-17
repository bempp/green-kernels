//! Implementation of the Laplace kernel
use crate::helpers::{
    check_dimensions_assemble, check_dimensions_assemble_diagonal, check_dimensions_evaluate,
};
use crate::traits::Kernel;
use crate::types::EvalType;
use num::traits::FloatConst;
use rayon::prelude::*;
use rlst::{RlstScalar, RlstSimd, SimdFor};
use std::marker::PhantomData;

/// Kernel for Laplace in 3D
#[derive(Clone, Default)]
pub struct Laplace3dKernel<T: RlstScalar> {
    _phantom_t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> Laplace3dKernel<T> {
    /// Create new
    pub fn new() -> Self {
        Self {
            _phantom_t: PhantomData,
        }
    }
}

impl<T: RlstScalar + Send + Sync> Kernel for Laplace3dKernel<T>
where
    <T as RlstScalar>::Real: Send + Sync,
{
    type T = T;

    fn domain_component_count(&self) -> usize {
        1
    }

    fn space_dimension(&self) -> usize {
        3
    }

    fn evaluate_st(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
    ) {
        check_dimensions_evaluate(self, eval_type, sources, targets, charges, result);
        let ntargets = targets.len() / self.space_dimension();
        let range_dim = self.range_component_count(eval_type);

        result
            .chunks_exact_mut(range_dim)
            .enumerate()
            .for_each(|(target_index, my_chunk)| {
                let target = [
                    targets[3 * target_index],
                    targets[3 * target_index + 1],
                    targets[3 * target_index + 2],
                ];

                evaluate_laplace_one_target(eval_type, &target, sources, charges, my_chunk)
            });
    }

    fn evaluate_mt(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
    ) {
        check_dimensions_evaluate(self, eval_type, sources, targets, charges, result);
        let ntargets = targets.len() / self.space_dimension();
        let range_dim = self.range_component_count(eval_type);

        result
            .par_chunks_exact_mut(range_dim)
            .enumerate()
            .for_each(|(target_index, my_chunk)| {
                let target = [
                    targets[3 * target_index],
                    targets[3 * target_index + 1],
                    targets[3 * target_index + 2],
                ];

                evaluate_laplace_one_target(eval_type, &target, sources, charges, my_chunk)
            });
    }

    fn assemble_st(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        check_dimensions_assemble(self, eval_type, sources, targets, result);
        let ntargets = targets.len() / self.space_dimension();
        let nsources = sources.len() / self.space_dimension();
        let range_dim = self.range_component_count(eval_type);

        result
            .chunks_exact_mut(range_dim * nsources)
            .enumerate()
            .for_each(|(target_index, my_chunk)| {
                let target = [
                    targets[target_index],
                    targets[ntargets + target_index],
                    targets[2 * ntargets + target_index],
                ];

                assemble_laplace_one_target(eval_type, &target, sources, my_chunk)
            });
    }

    fn assemble_mt(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        check_dimensions_assemble(self, eval_type, sources, targets, result);
        let ntargets = targets.len() / self.space_dimension();
        let nsources = sources.len() / self.space_dimension();
        let range_dim = self.range_component_count(eval_type);

        result
            .par_chunks_exact_mut(range_dim * nsources)
            .enumerate()
            .for_each(|(target_index, my_chunk)| {
                let target = [
                    targets[target_index],
                    targets[ntargets + target_index],
                    targets[2 * ntargets + target_index],
                ];

                assemble_laplace_one_target(eval_type, &target, sources, my_chunk)
            });
    }

    fn assemble_diagonal_st(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        check_dimensions_assemble_diagonal(self, eval_type, sources, targets, result);
        let ntargets = targets.len() / self.space_dimension();
        let range_dim = self.range_component_count(eval_type);

        result
            .chunks_exact_mut(range_dim)
            .enumerate()
            .for_each(|(target_index, my_chunk)| {
                let target = [
                    targets[target_index],
                    targets[ntargets + target_index],
                    targets[2 * ntargets + target_index],
                ];
                let source = [
                    sources[target_index],
                    sources[ntargets + target_index],
                    sources[2 * ntargets + target_index],
                ];
                self.greens_fct(eval_type, &source, &target, my_chunk)
            });
    }

    fn range_component_count(&self, eval_type: EvalType) -> usize {
        laplace_component_count(eval_type)
    }

    fn greens_fct(
        &self,
        eval_type: EvalType,
        source: &[<Self::T as RlstScalar>::Real],
        target: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        let zero_real = <T::Real as num::Zero>::zero();
        let one_real = <T::Real as num::One>::one();
        let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();
        let diff0 = source[0] - target[0];
        let diff1 = source[1] - target[1];
        let diff2 = source[2] - target[2];
        let diff_norm = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2).sqrt();
        let inv_diff_norm = {
            if diff_norm == zero_real {
                zero_real
            } else {
                one_real / diff_norm
            }
        };
        match eval_type {
            EvalType::Value => {
                result[0] = num::cast(inv_diff_norm * m_inv_4pi).unwrap();
            }
            EvalType::ValueDeriv => {
                let inv_diff_norm_cube = inv_diff_norm * inv_diff_norm * inv_diff_norm;
                result[0] = num::cast(inv_diff_norm * m_inv_4pi).unwrap();
                result[1] = num::cast(inv_diff_norm_cube * m_inv_4pi * diff0).unwrap();
                result[2] = num::cast(inv_diff_norm_cube * m_inv_4pi * diff1).unwrap();
                result[3] = num::cast(inv_diff_norm_cube * m_inv_4pi * diff2).unwrap();
            }
        }
    }
}

/// Evaluate laplce kernel with one target
pub fn evaluate_laplace_one_target<T: RlstScalar>(
    eval_type: EvalType,
    target: &[<T as RlstScalar>::Real],
    sources: &[<T as RlstScalar>::Real],
    charges: &[T],
    result: &mut [T],
) {
    let ncharges = charges.len();
    let nsources = ncharges;
    let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();
    let zero_real = <T::Real as num::Zero>::zero();
    let one_real = <T::Real as num::One>::one();

    let sources0 = &sources[0..nsources];
    let sources1 = &sources[nsources..2 * nsources];
    let sources2 = &sources[2 * nsources..3 * nsources];

    let mut diff0: T::Real;
    let mut diff1: T::Real;
    let mut diff2: T::Real;

    match eval_type {
        EvalType::Value => {
            struct Impl<'a, T: RlstScalar<Real = T> + RlstSimd> {
                t0: T,
                t1: T,
                t2: T,

                sources: &'a [T],
                charges: &'a [T],
            }

            impl<T: RlstScalar<Real = T> + RlstSimd> pulp::WithSimd for Impl<'_, T> {
                type Output = T;

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    use coe::Coerce;

                    let Self {
                        t0,
                        t1,
                        t2,
                        sources,
                        charges,
                    } = self;

                    let (sources, _) = pulp::as_arrays::<3, T>(sources);
                    let (sources_head, sources_tail) = T::as_simd_slice_from_vec(sources);
                    let (charges_head, charges_tail) = T::as_simd_slice(charges);

                    fn impl_slice<T: RlstScalar<Real = T> + RlstSimd, S: pulp::Simd>(
                        simd: S,
                        t0: T,
                        t1: T,
                        t2: T,
                        sources: &[[T::Scalars<S>; 3]],
                        charges: &[T::Scalars<S>],
                    ) -> T {
                        let simd = SimdFor::<T, S>::new(simd);

                        let t0 = simd.splat(t0);
                        let t1 = simd.splat(t1);
                        let t2 = simd.splat(t2);

                        let zero = simd.splat(T::zero());

                        let mut acc = simd.splat(T::zero());

                        for (&s, &c) in itertools::izip!(sources, charges) {
                            let [s0, s1, s2] = simd.deinterleave(s);

                            let diff0 = simd.sub(s0, t0);
                            let diff1 = simd.sub(s1, t1);
                            let diff2 = simd.sub(s2, t2);

                            let square_sum = simd.mul_add(
                                diff0,
                                diff0,
                                simd.mul_add(diff1, diff1, simd.mul(diff2, diff2)),
                            );

                            let is_zero = simd.cmp_eq(square_sum, zero);
                            let inv_abs =
                                simd.select(is_zero, zero, simd.approx_recip_sqrt(square_sum));

                            acc = simd.mul_add(inv_abs, c, acc);
                        }

                        simd.reduce_add(acc)
                    }

                    let acc0 = impl_slice::<T, S>(simd, t0, t1, t2, sources_head, charges_head);
                    let acc1 = impl_slice::<T, pulp::Scalar>(
                        pulp::Scalar::new(),
                        t0,
                        t1,
                        t2,
                        sources_tail.coerce(),
                        charges_tail.coerce(),
                    );

                    acc0 + acc1
                }
            }

            use coe::coerce_static as to;
            use coe::Coerce;
            if coe::is_same::<T, f32>() {
                let acc = pulp::Arch::new().dispatch(Impl::<'_, f32> {
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    charges: charges.coerce(),
                });
                result[0] += T::from_real(to::<_, T::Real>(acc)).mul_real(m_inv_4pi);
            } else if coe::is_same::<T, f64>() {
                let acc = pulp::Arch::new().dispatch(Impl::<'_, f64> {
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    charges: charges.coerce(),
                });
                result[0] += T::from_real(to::<_, T::Real>(acc)).mul_real(m_inv_4pi);
            } else {
                panic!()
            }
        }
        EvalType::ValueDeriv => {
            panic!("Not implemented.")
            //     // Cannot simply use an array my_result as this is not
            //     // correctly auto-vectorized.

            //     let mut my_result0 = T::zero();
            //     let mut my_result1 = T::zero();
            //     let mut my_result2 = T::zero();
            //     let mut my_result3 = T::zero();

            //     for index in 0..nsources {
            //         diff0 = sources0[index] - target[0];
            //         diff1 = sources1[index] - target[1];
            //         diff2 = sources2[index] - target[2];
            //         let diff_norm = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2).sqrt();
            //         let inv_diff_norm = {
            //             if diff_norm == zero_real {
            //                 zero_real
            //             } else {
            //                 one_real / diff_norm
            //             }
            //         };
            //         let inv_diff_norm_cubed = inv_diff_norm * inv_diff_norm * inv_diff_norm;

            //         my_result0 += charges[index].mul_real(inv_diff_norm);
            //         my_result1 += charges[index].mul_real(diff0 * inv_diff_norm_cubed);
            //         my_result2 += charges[index].mul_real(diff1 * inv_diff_norm_cubed);
            //         my_result3 += charges[index].mul_real(diff2 * inv_diff_norm_cubed);
            //     }

            //     result[0] += my_result0.mul_real(m_inv_4pi);
            //     result[1] += my_result1.mul_real(m_inv_4pi);
            //     result[2] += my_result2.mul_real(m_inv_4pi);
            //     result[3] += my_result3.mul_real(m_inv_4pi);
        }
    }
}

pub fn evaluate_laplace_one_target_f32(
    eval_type: EvalType,
    target: &[f32],
    sources: &[f32],
    charges: &[f32],
    result: &mut [f32],
) {
    evaluate_laplace_one_target(eval_type, target, sources, charges, result);
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
                use core::arch::aarch64;
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

/// Assemble Laplace kernel with one target
pub fn assemble_laplace_one_target<T: RlstScalar>(
    eval_type: EvalType,
    target: &[<T as RlstScalar>::Real],
    sources: &[<T as RlstScalar>::Real],
    result: &mut [T],
) {
    assert_eq!(sources.len() % 3, 0);
    assert_eq!(target.len(), 3);
    let nsources = sources.len() / 3;
    let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();
    let zero_real = <T::Real as num::Zero>::zero();
    let one_real = <T::Real as num::One>::one();

    let sources0 = &sources[0..nsources];
    let sources1 = &sources[nsources..2 * nsources];
    let sources2 = &sources[2 * nsources..3 * nsources];

    let mut diff0: T::Real;
    let mut diff1: T::Real;
    let mut diff2: T::Real;

    match eval_type {
        EvalType::Value => {
            let mut my_result;
            for index in 0..nsources {
                diff0 = sources0[index] - target[0];
                diff1 = sources1[index] - target[1];
                diff2 = sources2[index] - target[2];
                let diff_norm = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2).sqrt();
                let inv_diff_norm = {
                    if diff_norm == zero_real {
                        zero_real
                    } else {
                        one_real / diff_norm
                    }
                };

                my_result = inv_diff_norm * m_inv_4pi;
                result[index] = num::cast::<T::Real, T>(my_result).unwrap();
            }
        }
        EvalType::ValueDeriv => {
            // Cannot simply use an array my_result as this is not
            // correctly auto-vectorized.

            let mut my_result0;
            let mut my_result1;
            let mut my_result2;
            let mut my_result3;

            let mut chunks = result.chunks_exact_mut(nsources);

            let my_res0 = chunks.next().unwrap();
            let my_res1 = chunks.next().unwrap();
            let my_res2 = chunks.next().unwrap();
            let my_res3 = chunks.next().unwrap();

            for index in 0..nsources {
                //let my_res = &mut result[4 * index..4 * (index + 1)];
                diff0 = sources0[index] - target[0];
                diff1 = sources1[index] - target[1];
                diff2 = sources2[index] - target[2];
                let diff_norm = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2).sqrt();
                let inv_diff_norm = {
                    if diff_norm == zero_real {
                        zero_real
                    } else {
                        one_real / diff_norm
                    }
                };
                let inv_diff_norm_cubed = inv_diff_norm * inv_diff_norm * inv_diff_norm;

                my_result0 = T::one().mul_real(inv_diff_norm * m_inv_4pi);
                my_result1 = T::one().mul_real(diff0 * inv_diff_norm_cubed * m_inv_4pi);
                my_result2 = T::one().mul_real(diff1 * inv_diff_norm_cubed * m_inv_4pi);
                my_result3 = T::one().mul_real(diff2 * inv_diff_norm_cubed * m_inv_4pi);

                my_res0[index] = my_result0;
                my_res1[index] = my_result1;
                my_res2[index] = my_result2;
                my_res3[index] = my_result3;
            }
        }
    }
}

fn laplace_component_count(eval_type: EvalType) -> usize {
    match eval_type {
        EvalType::Value => 1,
        EvalType::ValueDeriv => 4,
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use approx::assert_relative_eq;
    use rand::prelude::*;

    use rlst::{
        rlst_dynamic_array1, rlst_dynamic_array2, Array, BaseArray, RandomAccessByRef,
        RandomAccessMut, RawAccess, RawAccessMut, Shape, VectorContainer,
    };

    fn copy(
        m_in: &Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    ) -> Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> {
        let mut m = rlst_dynamic_array2!(f64, m_in.shape());
        for i in 0..m_in.shape()[0] {
            for j in 0..m_in.shape()[1] {
                *m.get_mut([i, j]).unwrap() = *m_in.get([i, j]).unwrap();
            }
        }
        m
    }

    fn rand_mat(shape: [usize; 2]) -> Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> {
        let mut m = rlst_dynamic_array2!(f64, shape);
        let mut rng = rand::thread_rng();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                *m.get_mut([i, j]).unwrap() = rng.gen()
            }
        }
        m
    }

    fn rand_vec(size: usize) -> Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> {
        let mut v = rlst_dynamic_array2!(f64, [size, 1]);
        let mut rng = rand::thread_rng();
        for i in 0..size {
            *v.get_mut([i, 0]).unwrap() = rng.gen();
        }
        v
    }

    #[test]
    fn test_laplace_3d_value() {
        let nsources = 9;
        let ntargets = 2;

        let sources = rand_mat([3, nsources]);
        let targets = rand_mat([3, ntargets]);
        let charges = rand_vec(nsources);
        let mut green_value = rlst_dynamic_array2!(f64, [1, ntargets]);

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            charges.data(),
            green_value.data_mut(),
        );

        for target_index in 0..ntargets {
            let mut expected: f64 = 0.0;
            for source_index in 0..nsources {
                let dist = ((targets[[0, target_index]] - sources[[0, source_index]]).square()
                    + (targets[[1, target_index]] - sources[[1, source_index]]).square()
                    + (targets[[2, target_index]] - sources[[2, source_index]]).square())
                .sqrt();

                expected += charges[[source_index, 0]] * 0.25 * f64::FRAC_1_PI() / dist;
            }

            assert_relative_eq!(green_value[[0, target_index]], expected, epsilon = 1E-12);
        }
    }

    #[test]
    fn test_laplace_3d() {
        let eps: f64 = 1E-8;

        let nsources = 2;
        let ntargets = 1;

        let sources = rand_mat([nsources, 3]);
        let targets = rand_mat([ntargets, 3]);
        let charges = rand_vec(nsources);
        let mut green_value = rlst_dynamic_array2!(f64, [ntargets, 1]);

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            charges.data(),
            green_value.data_mut(),
        );

        for target_index in 0..ntargets {
            let mut expected: f64 = 0.0;
            for source_index in 0..nsources {
                let dist = ((targets[[target_index, 0]] - sources[[source_index, 0]]).square()
                    + (targets[[target_index, 1]] - sources[[source_index, 1]]).square()
                    + (targets[[target_index, 2]] - sources[[source_index, 2]]).square())
                .sqrt();

                expected += charges[[source_index, 0]] * 0.25 * f64::FRAC_1_PI() / dist;
            }

            assert_relative_eq!(green_value[[target_index, 0]], expected, epsilon = 1E-12);
        }

        let mut targets_x_eps = copy(&targets);
        let mut targets_y_eps = copy(&targets);
        let mut targets_z_eps = copy(&targets);

        for index in 0..ntargets {
            targets_x_eps[[index, 0]] += eps;
            targets_y_eps[[index, 1]] += eps;
            targets_z_eps[[index, 2]] += eps;
        }

        let mut expected = rlst_dynamic_array2!(f64, [4, ntargets]);

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            charges.data(),
            expected.data_mut(),
        );

        let mut green_value_x_eps = rlst_dynamic_array2![f64, [ntargets, 1]];

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_x_eps.data(),
            charges.data(),
            green_value_x_eps.data_mut(),
        );

        let mut green_value_y_eps = rlst_dynamic_array2![f64, [ntargets, 1]];

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_y_eps.data(),
            charges.data(),
            green_value_y_eps.data_mut(),
        );
        let mut green_value_z_eps = rlst_dynamic_array2![f64, [ntargets, 1]];

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_z_eps.data(),
            charges.data(),
            green_value_z_eps.data_mut(),
        );

        let gv0 = copy(&green_value);
        let gv1 = copy(&green_value);
        let gv2 = copy(&green_value);

        let mut x_deriv = rlst_dynamic_array2![f64, [ntargets, 1]];
        let mut y_deriv = rlst_dynamic_array2![f64, [ntargets, 1]];
        let mut z_deriv = rlst_dynamic_array2![f64, [ntargets, 1]];
        x_deriv.fill_from((green_value_x_eps - gv0) * (1.0 / eps));
        y_deriv.fill_from((green_value_y_eps - gv1) * (1.0 / eps));
        z_deriv.fill_from((green_value_z_eps - gv2) * (1.0 / eps));

        for target_index in 0..ntargets {
            assert_relative_eq!(
                green_value[[target_index, 0]],
                expected[[0, target_index]],
                epsilon = 1E-12
            );

            assert_relative_eq!(
                x_deriv[[target_index, 0]],
                expected[[1, target_index]],
                epsilon = 1E-5
            );
            assert_relative_eq!(
                y_deriv[[target_index, 0]],
                expected[[2, target_index]],
                epsilon = 1E-5
            );

            assert_relative_eq!(
                z_deriv[[target_index, 0]],
                expected[[3, target_index]],
                epsilon = 1E-5
            );
        }
    }

    #[test]
    fn test_assemble_laplace_3d() {
        let nsources = 3;
        let ntargets = 5;

        let sources = rand_mat([nsources, 3]);
        let targets = rand_mat([ntargets, 3]);
        let mut green_value_t = rlst_dynamic_array2!(f64, [nsources, ntargets]);

        Laplace3dKernel::<f64>::default().assemble_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            green_value_t.data_mut(),
        );

        // The matrix needs to be transposed so that the first row corresponds to the first target,
        // second row to the second target and so on.

        let mut green_value = rlst_dynamic_array2!(f64, [ntargets, nsources]);
        green_value.fill_from(green_value_t.transpose());

        for charge_index in 0..nsources {
            let mut charges = rlst_dynamic_array2![f64, [nsources, 1]];
            let mut expected = rlst_dynamic_array2![f64, [ntargets, 1]];
            charges[[charge_index, 0]] = 1.0;

            Laplace3dKernel::<f64>::default().evaluate_st(
                EvalType::Value,
                sources.data(),
                targets.data(),
                charges.data(),
                expected.data_mut(),
            );

            for target_index in 0..ntargets {
                assert_relative_eq!(
                    green_value[[target_index, charge_index]],
                    expected[[target_index, 0]],
                    epsilon = 1E-12
                );
            }
        }

        let mut green_value_deriv_t = rlst_dynamic_array2!(f64, [nsources, 4 * ntargets]);

        Laplace3dKernel::<f64>::default().assemble_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            green_value_deriv_t.data_mut(),
        );

        // The matrix needs to be transposed so that the first row corresponds to the first target, etc.

        let mut green_value_deriv = rlst_dynamic_array2!(f64, [4 * ntargets, nsources]);
        green_value_deriv.fill_from(green_value_deriv_t.transpose());

        for charge_index in 0..nsources {
            let mut charges = rlst_dynamic_array2![f64, [nsources, 1]];
            let mut expected = rlst_dynamic_array2!(f64, [4, ntargets]);

            charges[[charge_index, 0]] = 1.0;

            Laplace3dKernel::<f64>::default().evaluate_st(
                EvalType::ValueDeriv,
                sources.data(),
                targets.data(),
                charges.data(),
                expected.data_mut(),
            );

            for deriv_index in 0..4 {
                for target_index in 0..ntargets {
                    assert_relative_eq!(
                        green_value_deriv[[4 * target_index + deriv_index, charge_index]],
                        expected[[deriv_index, target_index]],
                        epsilon = 1E-12
                    );
                }
            }
        }
    }

    #[test]
    fn test_compare_assemble_with_direct_computation() {
        let nsources = 3;
        let ntargets = 5;

        let sources = rand_mat([nsources, 3]);
        let targets = rand_mat([ntargets, 3]);
        let mut green_value_deriv = rlst_dynamic_array2!(f64, [nsources, 4 * ntargets]);

        Laplace3dKernel::<f64>::default().assemble_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            green_value_deriv.data_mut(),
        );
    }

    #[test]
    fn test_assemble_diag_laplace_3d() {
        let nsources = 5;
        let ntargets = 5;

        let mut sources = rlst_dynamic_array2!(f64, [nsources, 3]);
        let mut targets = rlst_dynamic_array2!(f64, [ntargets, 3]);

        sources.fill_from_seed_equally_distributed(1);
        targets.fill_from_seed_equally_distributed(2);

        let mut green_value_diag = rlst_dynamic_array1!(f64, [ntargets]);
        let mut green_value_diag_deriv = rlst_dynamic_array2!(f64, [4, ntargets]);

        Laplace3dKernel::<f64>::default().assemble_diagonal_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            green_value_diag.data_mut(),
        );
        Laplace3dKernel::<f64>::default().assemble_diagonal_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            green_value_diag_deriv.data_mut(),
        );

        let mut green_value_t = rlst_dynamic_array2!(f64, [nsources, ntargets]);

        Laplace3dKernel::<f64>::default().assemble_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            green_value_t.data_mut(),
        );

        // The matrix needs to be transposed so that the first row corresponds to the first target,
        // second row to the second target and so on.

        let mut green_value = rlst_dynamic_array2!(f64, [ntargets, nsources]);
        green_value.fill_from(green_value_t.transpose());

        let mut green_value_deriv_t = rlst_dynamic_array2!(f64, [nsources, 4 * ntargets]);

        Laplace3dKernel::<f64>::default().assemble_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            green_value_deriv_t.data_mut(),
        );

        // The matrix needs to be transposed so that the first row corresponds to the first target, etc.

        let mut green_value_deriv = rlst_dynamic_array2!(f64, [4 * ntargets, nsources]);
        green_value_deriv.fill_from(green_value_deriv_t.transpose());

        for index in 0..nsources {
            assert_relative_eq!(
                green_value[[index, index]],
                green_value_diag[[index]],
                epsilon = 1E-12
            );

            assert_relative_eq!(
                green_value_deriv[[4 * index, index]],
                green_value_diag_deriv[[0, index]],
                epsilon = 1E-12,
            );

            assert_relative_eq!(
                green_value_deriv[[4 * index + 1, index]],
                green_value_diag_deriv[[1, index]],
                epsilon = 1E-12,
            );

            assert_relative_eq!(
                green_value_deriv[[4 * index + 2, index]],
                green_value_diag_deriv[[2, index]],
                epsilon = 1E-12,
            );

            assert_relative_eq!(
                green_value_deriv[[4 * index + 3, index]],
                green_value_diag_deriv[[3, index]],
                epsilon = 1E-12,
            );
        }
    }
}
