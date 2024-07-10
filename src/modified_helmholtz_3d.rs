//! Implementation of the Laplace kernel
use crate::helpers::{
    check_dimensions_assemble, check_dimensions_assemble_diagonal, check_dimensions_evaluate,
};
use crate::traits::Kernel;
use crate::types::EvalType;
use num::traits::FloatConst;
use rayon::prelude::*;
use rlst::{RlstScalar, RlstSimd, SimdFor};

/// Kernel for Modified Helmholtz in 3D
#[derive(Clone, Copy)]
pub struct ModifiedHelmholtz3dKernel<T: RlstScalar<Real = T>> {
    omega: T,
}

impl<T: RlstScalar<Real = T>> ModifiedHelmholtz3dKernel<T> {
    /// Create new
    pub fn new(omega: T) -> Self {
        Self { omega }
    }
}

impl<T: RlstScalar<Real = T> + Send + Sync> Kernel for ModifiedHelmholtz3dKernel<T> {
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

                evaluate_modified_helmholtz_one_target(
                    eval_type, &target, sources, charges, self.omega, my_chunk,
                )
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

                evaluate_modified_helmholtz_one_target(
                    eval_type, &target, sources, charges, self.omega, my_chunk,
                )
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

    fn assemble_pairwise_st(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        check_dimensions_assemble_diagonal(self, eval_type, sources, targets, result);
        let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();

        match eval_type {
            EvalType::Value => {
                struct Impl<'a, T: RlstScalar<Real = T> + RlstSimd> {
                    m_inv_4pi: T,
                    omega: T,

                    sources: &'a [T],
                    targets: &'a [T],
                    result: &'a mut [T],
                }

                impl<T: RlstScalar<Real = T> + RlstSimd> pulp::WithSimd for Impl<'_, T> {
                    type Output = ();

                    #[inline(always)]
                    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                        use coe::Coerce;

                        let Self {
                            m_inv_4pi,
                            omega,
                            sources,
                            targets,
                            result,
                        } = self;

                        let (sources, _) = pulp::as_arrays::<3, T>(sources);
                        let (targets, _) = pulp::as_arrays::<3, T>(targets);
                        let (sources_head, sources_tail) = T::as_simd_slice_from_vec(sources);
                        let (targets_head, targets_tail) = T::as_simd_slice_from_vec(targets);
                        let (result_head, result_tail) = T::as_simd_slice_mut(result);

                        fn impl_slice<T: RlstScalar<Real = T> + RlstSimd, S: pulp::Simd>(
                            simd: S,
                            m_inv_4pi: T,
                            omega: T,
                            sources: &[[T::Scalars<S>; 3]],
                            targets: &[[T::Scalars<S>; 3]],
                            result: &mut [T::Scalars<S>],
                        ) {
                            let simd = SimdFor::<T, S>::new(simd);

                            let m_inv_4pi = simd.splat(m_inv_4pi);
                            let zero = simd.splat(T::zero());

                            for (&s, &t, r) in itertools::izip!(sources, targets, result) {
                                let [s0, s1, s2] = simd.deinterleave(s);
                                let [t0, t1, t2] = simd.deinterleave(t);

                                let diff0 = simd.sub(s0, t0);
                                let diff1 = simd.sub(s1, t1);
                                let diff2 = simd.sub(s2, t2);

                                let square_sum = simd.mul_add(
                                    diff0,
                                    diff0,
                                    simd.mul_add(diff1, diff1, simd.mul(diff2, diff2)),
                                );

                                let is_zero = simd.cmp_eq(square_sum, zero);
                                let inv_diff_norm =
                                    simd.select(is_zero, zero, simd.approx_recip_sqrt(square_sum));

                                let diff_norm = simd.mul(inv_diff_norm, square_sum);

                                let romega = simd.mul(simd.splat(omega), diff_norm);

                                *r = simd.mul(
                                    simd.exp(simd.neg(romega)),
                                    simd.mul(inv_diff_norm, m_inv_4pi),
                                );
                            }
                        }

                        impl_slice::<T, S>(
                            simd,
                            m_inv_4pi,
                            omega,
                            sources_head,
                            targets_head,
                            result_head,
                        );
                        impl_slice::<T, pulp::Scalar>(
                            pulp::Scalar::new(),
                            m_inv_4pi,
                            omega,
                            sources_tail.coerce(),
                            targets_tail.coerce(),
                            result_tail.coerce(),
                        );
                    }
                }

                use coe::coerce_static as to;
                use coe::Coerce;
                if coe::is_same::<T, f32>() {
                    pulp::Arch::new().dispatch(Impl::<'_, f32> {
                        m_inv_4pi: to(m_inv_4pi),
                        omega: coe::coerce_static(self.omega),
                        sources: sources.coerce(),
                        targets: targets.coerce(),
                        result: result.coerce(),
                    });
                } else if coe::is_same::<T, f64>() {
                    pulp::Arch::new().dispatch(Impl::<'_, f64> {
                        m_inv_4pi: to(m_inv_4pi),
                        omega: coe::coerce_static(self.omega),
                        sources: sources.coerce(),
                        targets: targets.coerce(),
                        result: result.coerce(),
                    });
                } else {
                    panic!()
                }
            }
            EvalType::ValueDeriv => {
                struct Impl<'a, T: RlstScalar<Real = T> + RlstSimd> {
                    m_inv_4pi: T,
                    omega: T,

                    sources: &'a [T],
                    targets: &'a [T],
                    result: &'a mut [T],
                }

                impl<T: RlstScalar<Real = T> + RlstSimd> pulp::WithSimd for Impl<'_, T> {
                    type Output = ();

                    #[inline(always)]
                    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                        use coe::Coerce;

                        let Self {
                            m_inv_4pi,
                            omega,
                            sources,
                            targets,
                            result,
                        } = self;

                        let (sources, _) = pulp::as_arrays::<3, T>(sources);
                        let (sources_head, sources_tail) = T::as_simd_slice_from_vec(sources);
                        let (targets, _) = pulp::as_arrays::<3, T>(targets);
                        let (targets_head, targets_tail) = T::as_simd_slice_from_vec(targets);
                        let (result, _) = pulp::as_arrays_mut::<4, T>(result);
                        let (result_head, result_tail) =
                            T::as_simd_slice_from_vec_mut::<_, 4>(result);

                        fn impl_slice<T: RlstScalar<Real = T> + RlstSimd, S: pulp::Simd>(
                            simd: S,
                            m_inv_4pi: T,
                            omega: T,
                            sources: &[[T::Scalars<S>; 3]],
                            targets: &[[T::Scalars<S>; 3]],
                            result: &mut [[T::Scalars<S>; 4]],
                        ) {
                            let simd = SimdFor::<T, S>::new(simd);

                            let m_inv_4pi = simd.splat(m_inv_4pi);

                            let zero = simd.splat(T::zero());

                            for (&s, &t, r) in itertools::izip!(sources, targets, result) {
                                let [s0, s1, s2] = simd.deinterleave(s);
                                let [t0, t1, t2] = simd.deinterleave(t);

                                let diff0 = simd.sub(s0, t0);
                                let diff1 = simd.sub(s1, t1);
                                let diff2 = simd.sub(s2, t2);

                                let square_sum = simd.mul_add(
                                    diff0,
                                    diff0,
                                    simd.mul_add(diff1, diff1, simd.mul(diff2, diff2)),
                                );

                                let is_zero = simd.cmp_eq(square_sum, zero);
                                let inv_diff_norm =
                                    simd.select(is_zero, zero, simd.approx_recip_sqrt(square_sum));

                                let diff_norm = simd.mul(inv_diff_norm, square_sum);

                                let romega = simd.mul(simd.splat(omega), diff_norm);

                                let green = simd.mul(
                                    simd.exp(simd.neg(romega)),
                                    simd.mul(inv_diff_norm, m_inv_4pi),
                                );

                                let deriv_first_factor = simd.mul(
                                    simd.mul(green, simd.add(simd.splat(T::one()), romega)),
                                    simd.mul(inv_diff_norm, inv_diff_norm),
                                );

                                r[0] = green;
                                r[1] = simd.mul(diff0, deriv_first_factor);
                                r[2] = simd.mul(diff1, deriv_first_factor);
                                r[3] = simd.mul(diff2, deriv_first_factor);

                                *r = simd.interleave(*r);
                            }
                        }

                        impl_slice::<T, S>(
                            simd,
                            m_inv_4pi,
                            omega,
                            sources_head,
                            targets_head,
                            result_head,
                        );
                        impl_slice::<T, pulp::Scalar>(
                            pulp::Scalar::new(),
                            m_inv_4pi,
                            omega,
                            sources_tail.coerce(),
                            targets_tail.coerce(),
                            result_tail.coerce(),
                        );
                    }
                }

                use coe::coerce_static as to;
                use coe::Coerce;
                if coe::is_same::<T, f32>() {
                    pulp::Arch::new().dispatch(Impl::<'_, f32> {
                        m_inv_4pi: to(m_inv_4pi),
                        omega: to(self.omega),
                        sources: sources.coerce(),
                        targets: targets.coerce(),
                        result: result.coerce(),
                    });
                } else if coe::is_same::<T, f64>() {
                    pulp::Arch::new().dispatch(Impl::<'_, f64> {
                        m_inv_4pi: to(m_inv_4pi),
                        omega: to(self.omega),
                        sources: sources.coerce(),
                        targets: targets.coerce(),
                        result: result.coerce(),
                    });
                } else {
                    panic!()
                }
            }
        }
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

            let omega: f32 = coe::coerce_static(self.omega);

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

            let romega: f32 = omega * diff_norm;

            let inv_diff_norm = {
                if diff_norm == 0.0 {
                    0.0
                } else {
                    f32::recip(diff_norm)
                }
            };

            match eval_type {
                EvalType::Value => {
                    result[0] = coe::coerce_static(inv_diff_norm * m_inv_4pi * (-romega).exp());
                }
                EvalType::ValueDeriv => {
                    let inv_diff_norm_omega = inv_diff_norm * (-romega).exp() * m_inv_4pi;
                    let inv_diff_norm_cube_omega =
                        inv_diff_norm * inv_diff_norm * inv_diff_norm_omega;
                    result[0] = coe::coerce_static(inv_diff_norm_omega);
                    result[1] =
                        coe::coerce_static(inv_diff_norm_cube_omega * (1.0 + romega) * diff0);
                    result[2] =
                        coe::coerce_static(inv_diff_norm_cube_omega * (1.0 + romega) * diff1);
                    result[3] =
                        coe::coerce_static(inv_diff_norm_cube_omega * (1.0 + romega) * diff2);
                }
            }
        } else if coe::is_same::<Self::T, f64>() {
            coe::assert_same::<<Self::T as RlstScalar>::Real, f64>();

            let omega: f64 = coe::coerce_static(self.omega);

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

            let romega: f64 = omega * diff_norm;

            let inv_diff_norm = {
                if diff_norm == 0.0 {
                    0.0
                } else {
                    f64::recip(diff_norm)
                }
            };

            match eval_type {
                EvalType::Value => {
                    result[0] = coe::coerce_static(inv_diff_norm * m_inv_4pi * (-romega).exp());
                }
                EvalType::ValueDeriv => {
                    let inv_diff_norm_omega = inv_diff_norm * (-romega).exp() * m_inv_4pi;
                    let inv_diff_norm_cube_omega =
                        inv_diff_norm * inv_diff_norm * inv_diff_norm_omega;
                    result[0] = coe::coerce_static(inv_diff_norm_omega);
                    result[1] =
                        coe::coerce_static(inv_diff_norm_cube_omega * (1.0 + romega) * diff0);
                    result[2] =
                        coe::coerce_static(inv_diff_norm_cube_omega * (1.0 + romega) * diff1);
                    result[3] =
                        coe::coerce_static(inv_diff_norm_cube_omega * (1.0 + romega) * diff2);
                }
            }
        } else {
            panic!("Type not implemented.");
        }
    }
}

/// Evaluate laplce kernel with one target
pub fn evaluate_modified_helmholtz_one_target<T: RlstScalar>(
    eval_type: EvalType,
    target: &[<T as RlstScalar>::Real],
    sources: &[<T as RlstScalar>::Real],
    charges: &[T],
    omega: T,
    result: &mut [T],
) {
    let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();

    match eval_type {
        EvalType::Value => {
            struct Impl<'a, T: RlstScalar<Real = T> + RlstSimd> {
                omega: T,
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
                        omega,
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
                        omega: T,
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
                            let inv_diff_norm =
                                simd.select(is_zero, zero, simd.approx_recip_sqrt(square_sum));

                            let diff_norm = simd.mul(inv_diff_norm, square_sum);

                            let romega = simd.mul(simd.splat(omega), diff_norm);

                            let green = simd.mul(simd.exp(simd.neg(romega)), inv_diff_norm);

                            acc = simd.mul_add(green, c, acc);
                        }

                        simd.reduce_add(acc)
                    }

                    let acc0 =
                        impl_slice::<T, S>(simd, omega, t0, t1, t2, sources_head, charges_head);
                    let acc1 = impl_slice::<T, pulp::Scalar>(
                        pulp::Scalar::new(),
                        omega,
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
                    omega: to(omega),
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    charges: charges.coerce(),
                });
                result[0] += T::from_real(to::<_, T::Real>(acc)).mul_real(m_inv_4pi);
            } else if coe::is_same::<T, f64>() {
                let acc = pulp::Arch::new().dispatch(Impl::<'_, f64> {
                    omega: to(omega),
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
            struct Impl<'a, T: RlstScalar<Real = T> + RlstSimd> {
                omega: T,
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
                        omega,
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
                        omega: T,
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
                            let inv_diff_norm =
                                simd.select(is_zero, zero, simd.approx_recip_sqrt(square_sum));

                            let diff_norm = simd.mul(inv_diff_norm, square_sum);

                            let romega = simd.mul(simd.splat(omega), diff_norm);

                            let green = simd.mul(simd.exp(simd.neg(romega)), inv_diff_norm);

                            let deriv_first_factor = simd.mul(
                                simd.mul(green, simd.add(simd.splat(T::one()), romega)),
                                simd.mul(inv_diff_norm, inv_diff_norm),
                            );

                            acc0 = simd.mul_add(green, c, acc0);
                            acc1 = simd.mul_add(diff0, simd.mul(c, deriv_first_factor), acc1);
                            acc2 = simd.mul_add(diff1, simd.mul(c, deriv_first_factor), acc2);
                            acc3 = simd.mul_add(diff2, simd.mul(c, deriv_first_factor), acc3);
                        }

                        [
                            simd.reduce_add(acc0),
                            simd.reduce_add(acc1),
                            simd.reduce_add(acc2),
                            simd.reduce_add(acc3),
                        ]
                    }

                    let acc0 =
                        impl_slice::<T, S>(simd, omega, t0, t1, t2, sources_head, charges_head);
                    let acc1 = impl_slice::<T, pulp::Scalar>(
                        pulp::Scalar::new(),
                        omega,
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
                    omega: to(omega),
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    charges: charges.coerce(),
                });
                result[0] += T::from_real(to::<_, T::Real>(acc[0])).mul_real(m_inv_4pi);
                result[1] += T::from_real(to::<_, T::Real>(acc[1])).mul_real(m_inv_4pi);
                result[2] += T::from_real(to::<_, T::Real>(acc[2])).mul_real(m_inv_4pi);
                result[3] += T::from_real(to::<_, T::Real>(acc[3])).mul_real(m_inv_4pi);
            } else if coe::is_same::<T, f64>() {
                let acc = pulp::Arch::new().dispatch(Impl::<'_, f64> {
                    omega: to(omega),
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    charges: charges.coerce(),
                });
                result[0] += T::from_real(to::<_, T::Real>(acc[0])).mul_real(m_inv_4pi);
                result[1] += T::from_real(to::<_, T::Real>(acc[1])).mul_real(m_inv_4pi);
                result[2] += T::from_real(to::<_, T::Real>(acc[2])).mul_real(m_inv_4pi);
                result[3] += T::from_real(to::<_, T::Real>(acc[3])).mul_real(m_inv_4pi);
            } else {
                panic!()
            }
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

    use crate::laplace_3d::Laplace3dKernel;

    use super::*;
    use approx::assert_relative_eq;
    use itertools::izip;
    use paste::paste;
    use rand::prelude::*;
    use rlst::prelude::*;

    use rlst::rlst_dynamic_array1;

    macro_rules! impl_modified_helmholtz_tests {
        ($scalar:ty, $delta:expr, $deriv_eps:expr, $eps:expr) => {
            paste! {

                    #[test]
                    fn [<test_modified_helmholtz_green_ $scalar>]() {
                        let delta = $delta;
                        let deriv_eps = $deriv_eps;
                        let eps = $eps;

                        let omega = 1.5;

                        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
                        let mut source = rlst_dynamic_array1!($scalar, [3]);
                        let mut target = rlst_dynamic_array1!($scalar, [3]);

                        source.fill_from_equally_distributed(&mut rng);
                        target.fill_from_equally_distributed(&mut rng);

                        let mut actual_value = [0.0];
                        let mut actual_deriv = [0.0; 4];
                        let mut expected_deriv_x = [0.0];
                        let mut expected_deriv_y = [0.0];
                        let mut expected_deriv_z = [0.0];

                        ModifiedHelmholtz3dKernel::<$scalar>::new(omega).greens_fct(
                            EvalType::Value,
                            source.data(),
                            target.data(),
                            actual_value.as_mut_slice(),
                        );

                        ModifiedHelmholtz3dKernel::<$scalar>::new(omega).greens_fct(
                            EvalType::ValueDeriv,
                            source.data(),
                            target.data(),
                            actual_deriv.as_mut_slice(),
                        );

                        let mut target_x = rlst_dynamic_array1!($scalar, [3]);
                        let mut target_y = rlst_dynamic_array1!($scalar, [3]);
                        let mut target_z = rlst_dynamic_array1!($scalar, [3]);

                        target_x.fill_from(target.view());
                        target_y.fill_from(target.view());
                        target_z.fill_from(target.view());

                        target_x[[0]] += delta;
                        target_y[[1]] += delta;
                        target_z[[2]] += delta;

                        ModifiedHelmholtz3dKernel::<$scalar>::new(omega).greens_fct(
                            EvalType::Value,
                            source.data(),
                            target_x.data(),
                            expected_deriv_x.as_mut_slice(),
                        );

                        ModifiedHelmholtz3dKernel::<$scalar>::new(omega).greens_fct(
                            EvalType::Value,
                            source.data(),
                            target_y.data(),
                            expected_deriv_y.as_mut_slice(),
                        );

                        ModifiedHelmholtz3dKernel::<$scalar>::new(omega).greens_fct(
                            EvalType::Value,
                            source.data(),
                            target_z.data(),
                            expected_deriv_z.as_mut_slice(),
                        );

                        expected_deriv_x[0] = (expected_deriv_x[0] - actual_value[0]) / delta;
                        expected_deriv_y[0] = (expected_deriv_y[0] - actual_value[0]) / delta;
                        expected_deriv_z[0] = (expected_deriv_z[0] - actual_value[0]) / delta;

                        assert_relative_eq!(actual_value[0], actual_deriv[0], max_relative = eps);
                        assert_relative_eq!(
                            actual_deriv[1],
                            expected_deriv_x[0],
                            max_relative = deriv_eps
                        );
                        assert_relative_eq!(
                            actual_deriv[2],
                            expected_deriv_y[0],
                            max_relative = deriv_eps
                        );
                        assert_relative_eq!(
                            actual_deriv[3],
                            expected_deriv_z[0],
                            max_relative = deriv_eps
                        );
                    }

            #[test]
            fn [<test_modified_helmholtz_laplace_compare_ $scalar>]() {
                let eps = $eps;

                let omega = 0.0;

                let mut rng = rand::rngs::StdRng::seed_from_u64(0);
                let mut source = rlst_dynamic_array1!($scalar, [3]);
                let mut target = rlst_dynamic_array1!($scalar, [3]);

                source.fill_from_equally_distributed(&mut rng);
                target.fill_from_equally_distributed(&mut rng);

                let mut actual_deriv = [0.0; 4];
                let mut expected_deriv = [0.0; 4];

                ModifiedHelmholtz3dKernel::<$scalar>::new(omega).greens_fct(
                    EvalType::ValueDeriv,
                    source.data(),
                    target.data(),
                    actual_deriv.as_mut_slice(),
                );

                Laplace3dKernel::<$scalar>::new().greens_fct(
                    EvalType::ValueDeriv,
                    source.data(),
                    target.data(),
                    expected_deriv.as_mut_slice(),
                );

                for (a, e) in izip!(actual_deriv, expected_deriv) {
                    assert_relative_eq!(a, e, max_relative = eps);
                }
            }

            #[test]
            fn [<test_modified_helmholtz_assemble_pairwise_ $scalar>]() {
                let eps = $eps;
                let omega = 1.5;

                let npoints = 53;

                let mut rng = rand::rngs::StdRng::seed_from_u64(0);

                let mut sources = rlst_dynamic_array2!($scalar, [3, npoints]);
                let mut targets = rlst_dynamic_array2!($scalar, [3, npoints]);

                sources.fill_from_equally_distributed(&mut rng);
                targets.fill_from_equally_distributed(&mut rng);

                let mut result_value = rlst_dynamic_array1!($scalar, [npoints]);
                let mut result_value_deriv = rlst_dynamic_array2!($scalar, [4, npoints]);

                ModifiedHelmholtz3dKernel::<$scalar>::new(omega).assemble_pairwise_st(
                    EvalType::Value,
                    sources.data(),
                    targets.data(),
                    result_value.data_mut(),
                );

                ModifiedHelmholtz3dKernel::<$scalar>::new(omega).assemble_pairwise_st(
                    EvalType::ValueDeriv,
                    sources.data(),
                    targets.data(),
                    result_value_deriv.data_mut(),
                );

                for (s, t, res_value, res_deriv) in izip!(
                    sources.col_iter(),
                    targets.col_iter(),
                    result_value.iter(),
                    result_value_deriv.col_iter()
                ) {
                    let mut expected_val = [0.0; 1];
                    let mut expected_deriv = [0.0; 4];

                    ModifiedHelmholtz3dKernel::<$scalar>::new(omega).greens_fct(
                        EvalType::Value,
                        s.data(),
                        t.data(),
                        expected_val.as_mut_slice(),
                    );

                    ModifiedHelmholtz3dKernel::<$scalar>::new(omega).greens_fct(
                        EvalType::ValueDeriv,
                        s.data(),
                        t.data(),
                        expected_deriv.as_mut_slice(),
                    );

                    assert_relative_eq!(res_value, expected_val[0], max_relative = eps);

                    for (a, &e) in izip!(res_deriv.iter(), expected_val.iter()) {
                        assert_relative_eq!(a, e, max_relative = eps);
                    }
                }
            }


                    }
        };
    }
    impl_modified_helmholtz_tests!(f32, 1E-4, 1E-2, 1E-5);
    impl_modified_helmholtz_tests!(f64, 1E-8, 1E-4, 1E-13);

    #[test]
    fn test_modified_helmholtz_evaluate() {
        let eps = 1E-5;
        let omega = 1.5;

        let nsources = 53;
        let ntargets = 47;

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let mut sources = rlst_dynamic_array2!(f32, [3, nsources]);
        let mut targets = rlst_dynamic_array2!(f32, [3, ntargets]);
        let mut charges = rlst_dynamic_array1!(f32, [nsources]);

        sources.fill_from_equally_distributed(&mut rng);
        targets.fill_from_equally_distributed(&mut rng);
        charges.fill_from_equally_distributed(&mut rng);

        // Evaluate expected contribution.

        let mut expected_value = rlst_dynamic_array1!(f32, [ntargets]);
        let mut expected_value_deriv = rlst_dynamic_array2!(f32, [4, ntargets]);

        for (e_val, target, mut e_val_deriv) in izip!(
            expected_value.iter_mut(),
            targets.col_iter(),
            expected_value_deriv.col_iter_mut()
        ) {
            for (source, charge) in izip!(sources.col_iter(), charges.iter()) {
                let mut res_val = [0.0];
                let mut res_val_deriv = [0.0; 4];
                ModifiedHelmholtz3dKernel::<f32>::new(omega).greens_fct(
                    EvalType::Value,
                    source.data(),
                    target.data(),
                    res_val.as_mut_slice(),
                );
                ModifiedHelmholtz3dKernel::<f32>::new(omega).greens_fct(
                    EvalType::ValueDeriv,
                    source.data(),
                    target.data(),
                    res_val_deriv.as_mut_slice(),
                );

                *e_val += charge * res_val[0];

                for (&r, e) in izip!(res_val_deriv.iter(), e_val_deriv.iter_mut()) {
                    *e += charge * r;
                }
            }
        }

        // Now compute the actual contribution

        let mut actual_value = rlst_dynamic_array1!(f32, [ntargets]);
        let mut actual_value_deriv = rlst_dynamic_array2!(f32, [4, ntargets]);

        ModifiedHelmholtz3dKernel::<f32>::new(omega).evaluate_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            charges.data(),
            actual_value.data_mut(),
        );
        ModifiedHelmholtz3dKernel::<f32>::new(omega).evaluate_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            charges.data(),
            actual_value_deriv.data_mut(),
        );

        for (a, e) in izip!(actual_value.iter(), expected_value.iter()) {
            assert_relative_eq!(a, e, max_relative = eps);
        }
        for (a, e) in izip!(actual_value_deriv.iter(), expected_value_deriv.iter()) {
            assert_relative_eq!(a, e, max_relative = eps);
        }
    }
}
