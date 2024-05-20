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
                    targets[3 * target_index],
                    targets[3 * target_index + 1],
                    targets[3 * target_index + 2],
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
                    targets[3 * target_index],
                    targets[3 * target_index + 1],
                    targets[3 * target_index + 2],
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

    #[inline(always)]
    fn greens_fct(
        &self,
        eval_type: EvalType,
        source: &[<Self::T as RlstScalar>::Real],
        target: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        assert_eq!(source.len(), 3);
        assert_eq!(target.len(), 3);

        if coe::is_same::<<Self::T as RlstScalar>::Real, f32>() {
            coe::assert_same::<<Self::T as RlstScalar>::Real, f32>();

            let m_inv_4pi: f32 = 0.25 * f32::FRAC_1_PI();

            let source: &[f32] = coe::coerce(source);
            let target: &[f32] = coe::coerce(target);
            let source: &[f32; 3] = source.try_into().unwrap();
            let target: &[f32; 3] = target.try_into().unwrap();

            let diff0 = source[0] - target[0];
            let diff1 = source[1] - target[1];
            let diff2 = source[2] - target[2];
            let diff_norm =
                f32::mul_add(diff0, diff0, f32::mul_add(diff1, diff1, diff2 * diff2)).sqrt();

            let inv_diff_norm = {
                if diff_norm == 0.0 {
                    0.0
                } else {
                    f32::recip(diff_norm)
                }
            };

            match eval_type {
                EvalType::Value => {
                    result[0] = coe::coerce_static(inv_diff_norm * m_inv_4pi);
                }
                EvalType::ValueDeriv => {
                    let inv_diff_norm_cube = inv_diff_norm * inv_diff_norm * inv_diff_norm;
                    result[0] = coe::coerce_static(inv_diff_norm * m_inv_4pi);
                    result[1] = coe::coerce_static(inv_diff_norm_cube * m_inv_4pi * diff0);
                    result[2] = coe::coerce_static(inv_diff_norm_cube * m_inv_4pi * diff1);
                    result[3] = coe::coerce_static(inv_diff_norm_cube * m_inv_4pi * diff2);
                }
            }
        } else if coe::is_same::<Self::T, f64>() {
            coe::assert_same::<<Self::T as RlstScalar>::Real, f64>();

            let m_inv_4pi: f64 = 0.25 * f64::FRAC_1_PI();

            let source: &[f64] = coe::coerce(source);
            let target: &[f64] = coe::coerce(target);
            let source: &[f64; 3] = source.try_into().unwrap();
            let target: &[f64; 3] = target.try_into().unwrap();

            let diff0 = source[0] - target[0];
            let diff1 = source[1] - target[1];
            let diff2 = source[2] - target[2];
            let diff_norm =
                f64::mul_add(diff0, diff0, f64::mul_add(diff1, diff1, diff2 * diff2)).sqrt();

            let inv_diff_norm = {
                if diff_norm == 0.0 {
                    0.0
                } else {
                    f64::recip(diff_norm)
                }
            };

            match eval_type {
                EvalType::Value => {
                    result[0] = coe::coerce_static(inv_diff_norm * m_inv_4pi);
                }
                EvalType::ValueDeriv => {
                    let inv_diff_norm_cube = inv_diff_norm * inv_diff_norm * inv_diff_norm;
                    result[0] = coe::coerce_static(inv_diff_norm * m_inv_4pi);
                    result[1] = coe::coerce_static(inv_diff_norm_cube * m_inv_4pi * diff0);
                    result[2] = coe::coerce_static(inv_diff_norm_cube * m_inv_4pi * diff1);
                    result[3] = coe::coerce_static(inv_diff_norm_cube * m_inv_4pi * diff2);
                }
            }
        } else {
            panic!("Type not implemented.");
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
    let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();

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
                result[0] = T::from_real(to::<_, T::Real>(acc)).mul_real(m_inv_4pi);
            } else {
                panic!()
            }
        }
        EvalType::ValueDeriv => {
            struct Impl<'a, T: RlstScalar<Real = T> + RlstSimd> {
                t0: T,
                t1: T,
                t2: T,

                sources: &'a [T],
                charges: &'a [T],
            }

            impl<T: RlstScalar<Real = T> + RlstSimd> pulp::WithSimd for Impl<'_, T> {
                type Output = [T; 4];

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
                    ) -> [T; 4] {
                        let simd = SimdFor::<T, S>::new(simd);

                        let t0 = simd.splat(t0);
                        let t1 = simd.splat(t1);
                        let t2 = simd.splat(t2);

                        let zero = simd.splat(T::zero());

                        let mut acc0 = simd.splat(T::zero());
                        let mut acc1 = simd.splat(T::zero());
                        let mut acc2 = simd.splat(T::zero());
                        let mut acc3 = simd.splat(T::zero());

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

                            let inv_abs_cube = simd.mul(inv_abs, simd.mul(inv_abs, inv_abs));

                            acc0 = simd.mul_add(inv_abs, c, acc0);
                            acc1 = simd.mul_add(diff0, simd.mul(c, inv_abs_cube), acc1);
                            acc2 = simd.mul_add(diff1, simd.mul(c, inv_abs_cube), acc2);
                            acc3 = simd.mul_add(diff2, simd.mul(c, inv_abs_cube), acc3);
                        }

                        [
                            simd.reduce_add(acc0),
                            simd.reduce_add(acc1),
                            simd.reduce_add(acc2),
                            simd.reduce_add(acc3),
                        ]
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

                    [
                        acc0[0] + acc1[0],
                        acc0[1] + acc1[1],
                        acc0[2] + acc1[2],
                        acc0[3] + acc1[3],
                    ]
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
                result[0] = T::from_real(to::<_, T::Real>(acc[0])).mul_real(m_inv_4pi);
                result[1] = T::from_real(to::<_, T::Real>(acc[1])).mul_real(m_inv_4pi);
                result[2] = T::from_real(to::<_, T::Real>(acc[2])).mul_real(m_inv_4pi);
                result[3] = T::from_real(to::<_, T::Real>(acc[3])).mul_real(m_inv_4pi);
            } else if coe::is_same::<T, f64>() {
                let acc = pulp::Arch::new().dispatch(Impl::<'_, f64> {
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    charges: charges.coerce(),
                });
                result[0] = T::from_real(to::<_, T::Real>(acc[0])).mul_real(m_inv_4pi);
                result[1] = T::from_real(to::<_, T::Real>(acc[1])).mul_real(m_inv_4pi);
                result[2] = T::from_real(to::<_, T::Real>(acc[2])).mul_real(m_inv_4pi);
                result[3] = T::from_real(to::<_, T::Real>(acc[3])).mul_real(m_inv_4pi);
            } else {
                panic!()
            }
        }
    }
}

// pub fn evaluate_laplace_one_target_f32(
//     eval_type: EvalType,
//     target: &[f32],
//     sources: &[f32],
//     charges: &[f32],
//     result: &mut [f32],
// ) {
//     evaluate_laplace_one_target(eval_type, target, sources, charges, result);
// }

/// Assemble Laplace kernel with one target
pub fn assemble_laplace_one_target<T: RlstScalar>(
    eval_type: EvalType,
    target: &[<T as RlstScalar>::Real],
    sources: &[<T as RlstScalar>::Real],
    result: &mut [T],
) {
    assert_eq!(sources.len() % 3, 0);
    assert_eq!(target.len(), 3);
    let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();

    match eval_type {
        EvalType::Value => {
            struct Impl<'a, T: RlstScalar<Real = T> + RlstSimd> {
                m_inv_4pi: T,
                t0: T,
                t1: T,
                t2: T,

                sources: &'a [T],
                result: &'a mut [T],
            }

            impl<T: RlstScalar<Real = T> + RlstSimd> pulp::WithSimd for Impl<'_, T> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    use coe::Coerce;

                    let Self {
                        m_inv_4pi,
                        t0,
                        t1,
                        t2,
                        sources,
                        result,
                    } = self;

                    let (sources, _) = pulp::as_arrays::<3, T>(sources);
                    let (sources_head, sources_tail) = T::as_simd_slice_from_vec(sources);
                    let (result_head, result_tail) = T::as_simd_slice_mut(result);

                    fn impl_slice<T: RlstScalar<Real = T> + RlstSimd, S: pulp::Simd>(
                        simd: S,
                        m_inv_4pi: T,
                        t0: T,
                        t1: T,
                        t2: T,
                        sources: &[[T::Scalars<S>; 3]],
                        result: &mut [T::Scalars<S>],
                    ) {
                        let simd = SimdFor::<T, S>::new(simd);

                        let m_inv_4pi = simd.splat(m_inv_4pi);

                        let t0 = simd.splat(t0);
                        let t1 = simd.splat(t1);
                        let t2 = simd.splat(t2);

                        let zero = simd.splat(T::zero());

                        for (&s, r) in itertools::izip!(sources, result) {
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
                            *r = simd.select(
                                is_zero,
                                zero,
                                simd.mul(simd.approx_recip_sqrt(square_sum), m_inv_4pi),
                            );
                        }
                    }

                    impl_slice::<T, S>(simd, m_inv_4pi, t0, t1, t2, sources_head, result_head);
                    impl_slice::<T, pulp::Scalar>(
                        pulp::Scalar::new(),
                        m_inv_4pi,
                        t0,
                        t1,
                        t2,
                        sources_tail.coerce(),
                        result_tail.coerce(),
                    );
                }
            }

            use coe::coerce_static as to;
            use coe::Coerce;
            if coe::is_same::<T, f32>() {
                pulp::Arch::new().dispatch(Impl::<'_, f32> {
                    m_inv_4pi: to(m_inv_4pi),
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    result: result.coerce(),
                });
            } else if coe::is_same::<T, f64>() {
                let acc = pulp::Arch::new().dispatch(Impl::<'_, f64> {
                    m_inv_4pi: to(m_inv_4pi),
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    result: result.coerce(),
                });
            } else {
                panic!()
            }
        }
        EvalType::ValueDeriv => {
            struct Impl<'a, T: RlstScalar<Real = T> + RlstSimd> {
                m_inv_4pi: T,
                t0: T,
                t1: T,
                t2: T,

                sources: &'a [T],
                result: &'a mut [T],
            }

            impl<T: RlstScalar<Real = T> + RlstSimd> pulp::WithSimd for Impl<'_, T> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    use coe::Coerce;

                    let Self {
                        m_inv_4pi,
                        t0,
                        t1,
                        t2,
                        sources,
                        result,
                    } = self;

                    let (sources, _) = pulp::as_arrays::<3, T>(sources);
                    let (sources_head, sources_tail) = T::as_simd_slice_from_vec(sources);
                    let (result, _) = pulp::as_arrays_mut::<4, T>(result);
                    let (result_head, result_tail) = T::as_simd_slice_from_vec_mut::<_, 4>(result);

                    fn impl_slice<T: RlstScalar<Real = T> + RlstSimd, S: pulp::Simd>(
                        simd: S,
                        m_inv_4pi: T,
                        t0: T,
                        t1: T,
                        t2: T,
                        sources: &[[T::Scalars<S>; 3]],
                        result: &mut [[T::Scalars<S>; 4]],
                    ) {
                        let simd = SimdFor::<T, S>::new(simd);

                        let m_inv_4pi = simd.splat(m_inv_4pi);

                        let t0 = simd.splat(t0);
                        let t1 = simd.splat(t1);
                        let t2 = simd.splat(t2);

                        let zero = simd.splat(T::zero());

                        for (&s, r) in itertools::izip!(sources, result) {
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

                            let inv_abs_cube = simd.mul(inv_abs, simd.mul(inv_abs, inv_abs));

                            r[0] = simd.mul(inv_abs, m_inv_4pi);
                            r[1] = simd.mul(diff0, simd.mul(inv_abs_cube, m_inv_4pi));
                            r[2] = simd.mul(diff1, simd.mul(inv_abs_cube, m_inv_4pi));
                            r[3] = simd.mul(diff2, simd.mul(inv_abs_cube, m_inv_4pi));

                            *r = simd.interleave(*r);
                        }
                    }

                    // impl_slice::<T, S>(simd, m_inv_4pi, t0, t1, t2, sources_head, result_head);
                    impl_slice::<T, S>(simd, m_inv_4pi, t0, t1, t2, sources_head, result_head);
                    impl_slice::<T, pulp::Scalar>(
                        pulp::Scalar::new(),
                        m_inv_4pi,
                        t0,
                        t1,
                        t2,
                        sources_tail.coerce(),
                        result_tail.coerce(),
                    );
                }
            }

            use coe::coerce_static as to;
            use coe::Coerce;
            if coe::is_same::<T, f32>() {
                pulp::Arch::new().dispatch(Impl::<'_, f32> {
                    m_inv_4pi: to(m_inv_4pi),
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    result: result.coerce(),
                });
            } else if coe::is_same::<T, f64>() {
                pulp::Arch::new().dispatch(Impl::<'_, f64> {
                    m_inv_4pi: to(m_inv_4pi),
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    result: result.coerce(),
                });
            } else {
                panic!()
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
    use rlst::{assert_array_relative_eq, prelude::*};

    use rlst::{rlst_dynamic_array1, rlst_dynamic_array2};

    #[test]
    fn test_laplace_3d_value_f32() {
        let nparticles = 13;

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let mut sources = rlst_dynamic_array2!(f32, [3, nparticles]);
        let mut targets = rlst_dynamic_array2!(f32, [3, nparticles]);
        let mut charges = rlst_dynamic_array1!(f32, [nparticles]);
        let mut green_value = rlst_dynamic_array1!(f32, [nparticles]);

        sources.fill_from_equally_distributed(&mut rng);
        targets.fill_from(sources.view());
        charges.fill_from_equally_distributed(&mut rng);

        Laplace3dKernel::<f32>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            charges.data(),
            green_value.data_mut(),
        );

        for target_index in 0..nparticles {
            let mut expected: f32 = 0.0;
            for source_index in 0..nparticles {
                let dist = ((targets[[0, target_index]] - sources[[0, source_index]]).square()
                    + (targets[[1, target_index]] - sources[[1, source_index]]).square()
                    + (targets[[2, target_index]] - sources[[2, source_index]]).square())
                .sqrt();

                if dist > 0.0 {
                    expected += charges[[source_index]] * 0.25 * f32::FRAC_1_PI() / dist;
                }
            }

            assert_relative_eq!(green_value[[target_index]], expected, epsilon = 5E-7);
        }
    }

    #[test]
    fn test_laplace_3d_value_f64() {
        let nparticles = 13;

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut sources = rlst_dynamic_array2!(f64, [3, nparticles]);
        let mut targets = rlst_dynamic_array2!(f64, [3, nparticles]);
        let mut charges = rlst_dynamic_array1!(f64, [nparticles]);
        let mut green_value = rlst_dynamic_array1!(f64, [nparticles]);

        sources.fill_from_equally_distributed(&mut rng);
        targets.fill_from(sources.view());
        charges.fill_from_equally_distributed(&mut rng);

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            charges.data(),
            green_value.data_mut(),
        );

        for target_index in 0..nparticles {
            let mut expected: f64 = 0.0;
            for source_index in 0..nparticles {
                let dist = ((targets[[0, target_index]] - sources[[0, source_index]]).square()
                    + (targets[[1, target_index]] - sources[[1, source_index]]).square()
                    + (targets[[2, target_index]] - sources[[2, source_index]]).square())
                .sqrt();

                if dist > 0.0 {
                    expected += charges[[source_index]] * 0.25 * f64::FRAC_1_PI() / dist;
                }
            }

            assert_relative_eq!(green_value[[target_index]], expected, epsilon = 1E-14);
        }
    }

    #[test]
    fn test_laplace_green_f32() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut source = rlst_dynamic_array1!(f32, [3]);
        let mut target = rlst_dynamic_array1!(f32, [3]);

        source.fill_from_equally_distributed(&mut rng);
        target.fill_from_equally_distributed(&mut rng);

        let charge: [f32; 1] = [1.0];

        let mut result = [0.0];
        let mut expect: [f32; 1] = [0.0];

        Laplace3dKernel::<f32>::default().greens_fct(
            EvalType::Value,
            source.data(),
            target.data(),
            result.as_mut_slice(),
        );

        Laplace3dKernel::<f32>::default().evaluate_st(
            EvalType::Value,
            source.data(),
            target.data(),
            charge.as_slice(),
            expect.as_mut_slice(),
        );

        assert_relative_eq!(result[0], expect[0], epsilon = 1E-5);

        let mut result = [0.0, 0.0, 0.0, 0.0];
        let mut expect: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

        Laplace3dKernel::<f32>::default().greens_fct(
            EvalType::ValueDeriv,
            source.data(),
            target.data(),
            result.as_mut_slice(),
        );

        Laplace3dKernel::<f32>::default().evaluate_st(
            EvalType::ValueDeriv,
            source.data(),
            target.data(),
            charge.as_slice(),
            expect.as_mut_slice(),
        );

        for index in 0..4 {
            assert_relative_eq!(result[index], expect[index], epsilon = 1E-5);
        }
    }

    #[test]
    fn test_laplace_green_f64() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut source = rlst_dynamic_array1!(f64, [3]);
        let mut target = rlst_dynamic_array1!(f64, [3]);

        source.fill_from_equally_distributed(&mut rng);
        target.fill_from_equally_distributed(&mut rng);

        let charge: [f64; 1] = [1.0];

        let mut result = [0.0];
        let mut expect: [f64; 1] = [0.0];

        Laplace3dKernel::<f64>::default().greens_fct(
            EvalType::Value,
            source.data(),
            target.data(),
            result.as_mut_slice(),
        );

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            source.data(),
            target.data(),
            charge.as_slice(),
            expect.as_mut_slice(),
        );

        assert_relative_eq!(result[0], expect[0], epsilon = 1E-12);

        let mut result = [0.0, 0.0, 0.0, 0.0];
        let mut expect: [f64; 4] = [0.0, 0.0, 0.0, 0.0];

        Laplace3dKernel::<f64>::default().greens_fct(
            EvalType::ValueDeriv,
            source.data(),
            target.data(),
            result.as_mut_slice(),
        );

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::ValueDeriv,
            source.data(),
            target.data(),
            charge.as_slice(),
            expect.as_mut_slice(),
        );

        for index in 0..4 {
            assert_relative_eq!(result[index], expect[index], epsilon = 1E-12);
        }
    }

    #[test]
    fn test_laplace_3d() {
        let eps: f64 = 1E-8;

        let nsources = 13;
        let ntargets = 7;

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut sources = rlst_dynamic_array2!(f64, [3, nsources]);
        let mut targets = rlst_dynamic_array2!(f64, [3, ntargets]);
        let mut charges = rlst_dynamic_array1!(f64, [nsources]);
        let mut green_value = rlst_dynamic_array1!(f64, [ntargets]);

        sources.fill_from_equally_distributed(&mut rng);
        targets.fill_from_equally_distributed(&mut rng);
        charges.fill_from_equally_distributed(&mut rng);

        Laplace3dKernel::default().evaluate_st(
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

                if dist > 0.0 {
                    expected += charges[[source_index]] * 0.25 * f64::FRAC_1_PI() / dist;
                }
            }

            assert_relative_eq!(green_value[[target_index]], expected, epsilon = 1E-12);
        }

        let mut targets_x_eps = rlst_dynamic_array2!(f64, [3, ntargets]);
        let mut targets_y_eps = rlst_dynamic_array2!(f64, [3, ntargets]);
        let mut targets_z_eps = rlst_dynamic_array2!(f64, [3, ntargets]);

        targets_x_eps.fill_from(targets.view());
        targets_y_eps.fill_from(targets.view());
        targets_z_eps.fill_from(targets.view());

        for index in 0..ntargets {
            targets_x_eps[[0, index]] += eps;
            targets_y_eps[[1, index]] += eps;
            targets_z_eps[[2, index]] += eps;
        }

        let mut expected = rlst_dynamic_array2!(f64, [4, ntargets]);

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            charges.data(),
            expected.data_mut(),
        );

        let mut green_value_x_eps = rlst_dynamic_array1![f64, [ntargets]];

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_x_eps.data(),
            charges.data(),
            green_value_x_eps.data_mut(),
        );

        let mut green_value_y_eps = rlst_dynamic_array1![f64, [ntargets]];

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_y_eps.data(),
            charges.data(),
            green_value_y_eps.data_mut(),
        );
        let mut green_value_z_eps = rlst_dynamic_array1![f64, [ntargets]];

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_z_eps.data(),
            charges.data(),
            green_value_z_eps.data_mut(),
        );

        let mut gv0 = rlst_dynamic_array1!(f64, [ntargets]);
        let mut gv1 = rlst_dynamic_array1!(f64, [ntargets]);
        let mut gv2 = rlst_dynamic_array1!(f64, [ntargets]);

        gv0.fill_from(green_value.view());
        gv1.fill_from(green_value.view());
        gv2.fill_from(green_value.view());

        let mut x_deriv = rlst_dynamic_array1![f64, [ntargets]];
        let mut y_deriv = rlst_dynamic_array1![f64, [ntargets]];
        let mut z_deriv = rlst_dynamic_array1![f64, [ntargets]];
        x_deriv.fill_from((green_value_x_eps - gv0).scalar_mul(1.0 / eps));
        y_deriv.fill_from((green_value_y_eps - gv1).scalar_mul(1.0 / eps));
        z_deriv.fill_from((green_value_z_eps - gv2).scalar_mul(1.0 / eps));

        for target_index in 0..ntargets {
            assert_relative_eq!(
                green_value[[target_index]],
                expected[[0, target_index]],
                epsilon = 1E-12
            );

            assert_relative_eq!(
                x_deriv[[target_index]],
                expected[[1, target_index]],
                epsilon = 1E-5
            );
            assert_relative_eq!(
                y_deriv[[target_index]],
                expected[[2, target_index]],
                epsilon = 1E-5
            );

            assert_relative_eq!(
                z_deriv[[target_index]],
                expected[[3, target_index]],
                epsilon = 1E-5
            );
        }
    }

    #[test]
    fn test_assemble_laplace_3d() {
        let nsources = 3;
        let ntargets = 1;

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut sources = rlst_dynamic_array2!(f64, [3, nsources]);
        let mut targets = rlst_dynamic_array2!(f64, [3, ntargets]);
        let mut green_value_t = rlst_dynamic_array2!(f64, [nsources, ntargets]);

        targets.fill_from_equally_distributed(&mut rng);
        sources.fill_from_equally_distributed(&mut rng);

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
            let mut charges = rlst_dynamic_array1![f64, [nsources]];
            let mut expected = rlst_dynamic_array1![f64, [ntargets]];
            charges[[charge_index]] = 1.0;

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
                    expected[[target_index]],
                    epsilon = 1E-12
                );
            }
        }

        let mut green_value_deriv = rlst_dynamic_array2!(f64, [4 * ntargets, nsources]);

        Laplace3dKernel::<f64>::default().assemble_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            green_value_deriv.data_mut(),
        );

        // The matrix needs to be transposed so that the first row corresponds to the first target, etc.

        for charge_index in 0..nsources {
            let mut charges = rlst_dynamic_array1![f64, [nsources]];
            let mut expected = rlst_dynamic_array2!(f64, [4, ntargets]);

            charges[[charge_index]] = 1.0;

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

    //     #[test]
    //     fn test_compare_assemble_with_direct_computation() {
    //         let nsources = 3;
    //         let ntargets = 5;

    //         let sources = rand_mat([nsources, 3]);
    //         let targets = rand_mat([ntargets, 3]);
    //         let mut green_value_deriv = rlst_dynamic_array2!(f64, [nsources, 4 * ntargets]);

    //         Laplace3dKernel::<f64>::default().assemble_st(
    //             EvalType::ValueDeriv,
    //             sources.data(),
    //             targets.data(),
    //             green_value_deriv.data_mut(),
    //         );
    //     }

    //     #[test]
    //     fn test_assemble_diag_laplace_3d() {
    //         let nsources = 5;
    //         let ntargets = 5;

    //         let mut sources = rlst_dynamic_array2!(f64, [nsources, 3]);
    //         let mut targets = rlst_dynamic_array2!(f64, [ntargets, 3]);

    //         sources.fill_from_seed_equally_distributed(1);
    //         targets.fill_from_seed_equally_distributed(2);

    //         let mut green_value_diag = rlst_dynamic_array1!(f64, [ntargets]);
    //         let mut green_value_diag_deriv = rlst_dynamic_array2!(f64, [4, ntargets]);

    //         Laplace3dKernel::<f64>::default().assemble_diagonal_st(
    //             EvalType::Value,
    //             sources.data(),
    //             targets.data(),
    //             green_value_diag.data_mut(),
    //         );
    //         Laplace3dKernel::<f64>::default().assemble_diagonal_st(
    //             EvalType::ValueDeriv,
    //             sources.data(),
    //             targets.data(),
    //             green_value_diag_deriv.data_mut(),
    //         );

    //         let mut green_value_t = rlst_dynamic_array2!(f64, [nsources, ntargets]);

    //         Laplace3dKernel::<f64>::default().assemble_st(
    //             EvalType::Value,
    //             sources.data(),
    //             targets.data(),
    //             green_value_t.data_mut(),
    //         );

    //         // The matrix needs to be transposed so that the first row corresponds to the first target,
    //         // second row to the second target and so on.

    //         let mut green_value = rlst_dynamic_array2!(f64, [ntargets, nsources]);
    //         green_value.fill_from(green_value_t.transpose());

    //         let mut green_value_deriv_t = rlst_dynamic_array2!(f64, [nsources, 4 * ntargets]);

    //         Laplace3dKernel::<f64>::default().assemble_st(
    //             EvalType::ValueDeriv,
    //             sources.data(),
    //             targets.data(),
    //             green_value_deriv_t.data_mut(),
    //         );

    //         // The matrix needs to be transposed so that the first row corresponds to the first target, etc.

    //         let mut green_value_deriv = rlst_dynamic_array2!(f64, [4 * ntargets, nsources]);
    //         green_value_deriv.fill_from(green_value_deriv_t.transpose());

    //         for index in 0..nsources {
    //             assert_relative_eq!(
    //                 green_value[[index, index]],
    //                 green_value_diag[[index]],
    //                 epsilon = 1E-12
    //             );

    //             assert_relative_eq!(
    //                 green_value_deriv[[4 * index, index]],
    //                 green_value_diag_deriv[[0, index]],
    //                 epsilon = 1E-12,
    //             );

    //             assert_relative_eq!(
    //                 green_value_deriv[[4 * index + 1, index]],
    //                 green_value_diag_deriv[[1, index]],
    //                 epsilon = 1E-12,
    //             );

    //             assert_relative_eq!(
    //                 green_value_deriv[[4 * index + 2, index]],
    //                 green_value_diag_deriv[[2, index]],
    //                 epsilon = 1E-12,
    //             );

    //             assert_relative_eq!(
    //                 green_value_deriv[[4 * index + 3, index]],
    //                 green_value_diag_deriv[[3, index]],
    //                 epsilon = 1E-12,
    //             );
    //         }
    //     }
}
