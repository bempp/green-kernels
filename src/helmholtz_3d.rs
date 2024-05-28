//! Implementation of the Helmholtz kernel
use crate::helpers::{
    check_dimensions_assemble, check_dimensions_assemble_diagonal, check_dimensions_evaluate,
};
use crate::traits::Kernel;
use crate::types::EvalType;
use num::traits::FloatConst;
use num::One;
use num::Zero;
use pulp::Simd;
use rayon::prelude::*;
use rlst::c32;
use rlst::c64;
use rlst::RlstScalar;
use rlst::{RlstSimd, SimdFor};
use std::marker::PhantomData;

/// Kernel for Helmholtz in 3D
#[derive(Clone, Default)]
pub struct Helmholtz3dKernel<T: RlstScalar> {
    /// Wavenumber
    pub wavenumber: T::Real,
    _phantom_t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> Helmholtz3dKernel<T> {
    /// Create new
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            wavenumber,
            _phantom_t: PhantomData,
        }
    }
}

impl<T: RlstScalar<Complex = T> + Send + Sync> Kernel for Helmholtz3dKernel<T>
where
    // Send and sync are defined for all the standard types that implement RlstScalar (f32, f64, c32, c64)
    <T as RlstScalar>::Complex: Send + Sync,
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

                evaluate_helmholtz_one_target(
                    eval_type,
                    &target,
                    sources,
                    charges,
                    self.wavenumber,
                    my_chunk,
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

                evaluate_helmholtz_one_target(
                    eval_type,
                    &target,
                    sources,
                    charges,
                    self.wavenumber,
                    my_chunk,
                )
            });
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

        let kr: T::Real = diff_norm * self.wavenumber;
        match eval_type {
            EvalType::Value => {
                result[0] = T::complex(kr.cos(), kr.sin()).mul_real(inv_diff_norm * m_inv_4pi)
            }
            EvalType::ValueDeriv => {
                let inv_diff_norm_squared = inv_diff_norm * inv_diff_norm;
                let gr = T::complex(kr.cos(), kr.sin()).mul_real(inv_diff_norm * m_inv_4pi);
                let gr_diff = gr.mul_real(inv_diff_norm_squared) * T::complex(one_real, -kr);

                result[0] = gr;
                result[1] = gr_diff.mul_real(diff0);
                result[2] = gr_diff.mul_real(diff1);
                result[3] = gr_diff.mul_real(diff2);
            }
        }
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

                assemble_helmholtz_one_target(
                    eval_type,
                    &target,
                    sources,
                    self.wavenumber,
                    my_chunk,
                )
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

                assemble_helmholtz_one_target(
                    eval_type,
                    &target,
                    sources,
                    self.wavenumber,
                    my_chunk,
                )
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
                    targets[3 * target_index],
                    targets[3 * target_index + 1],
                    targets[3 * target_index + 2],
                ];
                let source = [
                    sources[3 * target_index],
                    sources[3 * target_index + 1],
                    sources[3 * target_index + 2],
                ];
                self.greens_fct(eval_type, &source, &target, my_chunk)
            });
    }

    fn range_component_count(&self, eval_type: EvalType) -> usize {
        helmholtz_component_count(eval_type)
    }
}

/// Evaluate Helmholtz kernel for one target
pub fn evaluate_helmholtz_one_target<T: RlstScalar<Complex = T>>(
    eval_type: EvalType,
    target: &[T::Real],
    sources: &[T::Real],
    charges: &[T],
    wavenumber: T::Real,
    result: &mut [T],
) {
    let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();
    match eval_type {
        EvalType::Value => {
            struct Impl<'a, T: RlstScalar<Complex = T>>
            where
                T::Real: RlstSimd,
            {
                wavenumber: T::Real,
                t0: T::Real,
                t1: T::Real,
                t2: T::Real,

                sources: &'a [T::Real],
                charges: &'a [T],
            }

            impl<T: RlstScalar<Complex = T>> pulp::WithSimd for Impl<'_, T>
            where
                T::Real: RlstSimd,
            {
                type Output = (T::Real, T::Real);

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    use coe::Coerce;

                    let Self {
                        wavenumber,
                        t0,
                        t1,
                        t2,
                        sources,
                        charges,
                    } = self;

                    let (sources, _) = pulp::as_arrays::<3, T::Real>(sources);
                    let (sources_head, sources_tail) = <T::Real>::as_simd_slice_from_vec(sources);
                    let charges: &[[T::Real; 2]] = bytemuck::cast_slice(charges);
                    let (charges_head, charges_tail) = <T::Real>::as_simd_slice_from_vec(charges);

                    #[inline(always)]
                    fn impl_slice<T: RlstScalar<Complex = T>, S: Simd>(
                        simd: S,
                        wavenumber: T::Real,
                        t0: T::Real,
                        t1: T::Real,
                        t2: T::Real,
                        sources: &[[<T::Real as RlstSimd>::Scalars<S>; 3]],
                        charges: &[[<T::Real as RlstSimd>::Scalars<S>; 2]],
                    ) -> (T::Real, T::Real)
                    where
                        T::Real: RlstSimd,
                    {
                        let simd = SimdFor::<T::Real, S>::new(simd);

                        let t0 = simd.splat(t0);
                        let t1 = simd.splat(t1);
                        let t2 = simd.splat(t2);
                        let zero = simd.splat(T::Real::zero());
                        let wavenumber = simd.splat(wavenumber);
                        let mut acc_re = simd.splat(T::Real::zero());
                        let mut acc_im = simd.splat(T::Real::zero());

                        for (&s, &c) in itertools::izip!(sources, charges) {
                            let [sx, sy, sz] = simd.deinterleave(s);
                            let [c_re, c_im] = simd.deinterleave(c);

                            let diff0 = simd.sub(sx, t0);
                            let diff1 = simd.sub(sy, t1);
                            let diff2 = simd.sub(sz, t2);

                            let diff_norm = simd.sqrt(simd.mul_add(
                                diff0,
                                diff0,
                                simd.mul_add(diff1, diff1, simd.mul(diff2, diff2)),
                            ));

                            let is_zero = simd.cmp_eq(diff_norm, zero);
                            let inv_diff_norm = simd.select(
                                is_zero,
                                zero,
                                simd.div(simd.splat(T::Real::one()), diff_norm),
                            );
                            let kr = simd.mul(wavenumber, diff_norm);

                            let (g_re, g_im) = {
                                let (s, c) = simd.sin_cos(kr);
                                (simd.mul(c, inv_diff_norm), simd.mul(s, inv_diff_norm))
                            };

                            acc_re = simd.mul_add(
                                g_re,
                                c_re,
                                simd.mul_add(simd.neg(g_im), c_im, acc_re),
                            );
                            acc_im = simd.mul_add(g_re, c_im, simd.mul_add(g_im, c_re, acc_im));
                        }
                        (simd.reduce_add(acc_re), simd.reduce_add(acc_im))
                    }

                    let (re0, im0) = impl_slice::<T, S>(
                        simd,
                        wavenumber,
                        t0,
                        t1,
                        t2,
                        sources_head,
                        charges_head,
                    );
                    let (re1, im1) = impl_slice::<T, pulp::Scalar>(
                        pulp::Scalar::new(),
                        wavenumber,
                        t0,
                        t1,
                        t2,
                        sources_tail.coerce(),
                        charges_tail.coerce(),
                    );

                    (re0 + re1, im0 + im1)
                }
            }

            use coe::coerce_static as to;
            use coe::Coerce;
            if coe::is_same::<T, c32>() {
                let (re, im) = pulp::Arch::new().dispatch(Impl::<'_, c32> {
                    wavenumber: to(wavenumber),
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    charges: charges.coerce(),
                });
                result[0] += T::Complex::complex(to::<_, T::Real>(re), to::<_, T::Real>(im))
                    .mul_real(m_inv_4pi);
            } else if coe::is_same::<T, c64>() {
                let (re, im) = pulp::Arch::new().dispatch(Impl::<'_, c64> {
                    wavenumber: to(wavenumber),
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    charges: charges.coerce(),
                });
                result[0] += T::Complex::complex(to::<_, T::Real>(re), to::<_, T::Real>(im))
                    .mul_real(m_inv_4pi);
            } else {
                panic!()
            }
        }
        EvalType::ValueDeriv => {
            struct Impl<'a, T: RlstScalar<Complex = T>>
            where
                T::Real: RlstSimd,
            {
                wavenumber: T::Real,
                t0: T::Real,
                t1: T::Real,
                t2: T::Real,

                sources: &'a [T::Real],
                charges: &'a [T],
            }

            impl<T: RlstScalar<Complex = T>> pulp::WithSimd for Impl<'_, T>
            where
                T::Real: RlstSimd,
            {
                type Output = [[T::Real; 2]; 4];

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    use coe::Coerce;

                    let Self {
                        wavenumber,
                        t0,
                        t1,
                        t2,
                        sources,
                        charges,
                    } = self;

                    let (sources, _) = pulp::as_arrays::<3, T::Real>(sources);
                    let (sources_head, sources_tail) = <T::Real>::as_simd_slice_from_vec(sources);
                    let charges: &[[T::Real; 2]] = bytemuck::cast_slice(charges);
                    let (charges_head, charges_tail) = <T::Real>::as_simd_slice_from_vec(charges);

                    #[inline(always)]
                    fn impl_slice<T: RlstScalar<Complex = T>, S: Simd>(
                        simd: S,
                        wavenumber: T::Real,
                        t0: T::Real,
                        t1: T::Real,
                        t2: T::Real,
                        sources: &[[<T::Real as RlstSimd>::Scalars<S>; 3]],
                        charges: &[[<T::Real as RlstSimd>::Scalars<S>; 2]],
                    ) -> [[T::Real; 2]; 4]
                    where
                        T::Real: RlstSimd,
                    {
                        let simd = SimdFor::<T::Real, S>::new(simd);

                        let t0 = simd.splat(t0);
                        let t1 = simd.splat(t1);
                        let t2 = simd.splat(t2);
                        let zero = simd.splat(T::Real::zero());
                        let wavenumber = simd.splat(wavenumber);
                        let mut acc0_re = simd.splat(T::Real::zero());
                        let mut acc0_im = simd.splat(T::Real::zero());
                        let mut acc1_re = simd.splat(T::Real::zero());
                        let mut acc1_im = simd.splat(T::Real::zero());
                        let mut acc2_re = simd.splat(T::Real::zero());
                        let mut acc2_im = simd.splat(T::Real::zero());
                        let mut acc3_re = simd.splat(T::Real::zero());
                        let mut acc3_im = simd.splat(T::Real::zero());

                        for (&s, &c) in itertools::izip!(sources, charges) {
                            let [sx, sy, sz] = simd.deinterleave(s);
                            let [c_re, c_im] = simd.deinterleave(c);

                            let diff0 = simd.sub(sx, t0);
                            let diff1 = simd.sub(sy, t1);
                            let diff2 = simd.sub(sz, t2);

                            let diff_norm = simd.sqrt(simd.mul_add(
                                diff0,
                                diff0,
                                simd.mul_add(diff1, diff1, simd.mul(diff2, diff2)),
                            ));

                            let is_zero = simd.cmp_eq(diff_norm, zero);
                            let inv_diff_norm = simd.select(
                                is_zero,
                                zero,
                                simd.div(simd.splat(T::Real::one()), diff_norm),
                            );
                            let kr = simd.mul(wavenumber, diff_norm);

                            let (g_re, g_im) = {
                                let (s, c) = simd.sin_cos(kr);
                                (simd.mul(c, inv_diff_norm), simd.mul(s, inv_diff_norm))
                            };

                            // Multiply already with charges as need for function and derivatives.

                            let (g_re, g_im) = (
                                simd.mul_add(simd.neg(g_im), c_im, simd.mul(g_re, c_re)),
                                simd.mul_add(g_im, c_re, simd.mul(g_re, c_im)),
                            );

                            let inv_diff_norm_squared = simd.mul(inv_diff_norm, inv_diff_norm);

                            let (g_deriv_re, g_deriv_im) = (
                                simd.mul(g_re, inv_diff_norm_squared),
                                simd.mul(g_im, inv_diff_norm_squared),
                            );

                            let (g_deriv_re, g_deriv_im) = (
                                simd.mul_add(g_deriv_im, kr, g_deriv_re),
                                simd.mul_add(simd.neg(g_deriv_re), kr, g_deriv_im),
                            );

                            acc0_re = simd.add(acc0_re, g_re);
                            acc0_im = simd.add(acc0_im, g_im);

                            acc1_re = simd.mul_add(g_deriv_re, diff0, acc1_re);
                            acc1_im = simd.mul_add(g_deriv_im, diff0, acc1_im);

                            acc2_re = simd.mul_add(g_deriv_re, diff1, acc2_re);
                            acc2_im = simd.mul_add(g_deriv_im, diff1, acc2_im);

                            acc3_re = simd.mul_add(g_deriv_re, diff2, acc3_re);
                            acc3_im = simd.mul_add(g_deriv_im, diff2, acc3_im);
                        }
                        [
                            [simd.reduce_add(acc0_re), simd.reduce_add(acc0_im)],
                            [simd.reduce_add(acc1_re), simd.reduce_add(acc1_im)],
                            [simd.reduce_add(acc2_re), simd.reduce_add(acc2_im)],
                            [simd.reduce_add(acc3_re), simd.reduce_add(acc3_im)],
                        ]
                    }

                    let res1 = impl_slice::<T, S>(
                        simd,
                        wavenumber,
                        t0,
                        t1,
                        t2,
                        sources_head,
                        charges_head,
                    );
                    let res2 = impl_slice::<T, pulp::Scalar>(
                        pulp::Scalar::new(),
                        wavenumber,
                        t0,
                        t1,
                        t2,
                        sources_tail.coerce(),
                        charges_tail.coerce(),
                    );

                    [
                        [res1[0][0] + res2[0][0], res1[0][1] + res2[0][1]],
                        [res1[1][0] + res2[1][0], res1[1][1] + res2[1][1]],
                        [res1[2][0] + res2[2][0], res1[2][1] + res2[2][1]],
                        [res1[3][0] + res2[3][0], res1[3][1] + res2[3][1]],
                    ]
                }
            }

            use coe::coerce_static as to;
            use coe::Coerce;
            if coe::is_same::<T, c32>() {
                let res = pulp::Arch::new().dispatch(Impl::<'_, c32> {
                    wavenumber: to(wavenumber),
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    charges: charges.coerce(),
                });
                result[0] +=
                    T::Complex::complex(to::<_, T::Real>(res[0][0]), to::<_, T::Real>(res[0][1]))
                        .mul_real(m_inv_4pi);

                result[1] +=
                    T::Complex::complex(to::<_, T::Real>(res[1][0]), to::<_, T::Real>(res[1][1]))
                        .mul_real(m_inv_4pi);

                result[2] +=
                    T::Complex::complex(to::<_, T::Real>(res[2][0]), to::<_, T::Real>(res[2][1]))
                        .mul_real(m_inv_4pi);

                result[3] +=
                    T::Complex::complex(to::<_, T::Real>(res[3][0]), to::<_, T::Real>(res[3][1]))
                        .mul_real(m_inv_4pi);
            } else if coe::is_same::<T, c64>() {
                let res = pulp::Arch::new().dispatch(Impl::<'_, c64> {
                    wavenumber: to(wavenumber),
                    t0: to(target[0]),
                    t1: to(target[1]),
                    t2: to(target[2]),
                    sources: sources.coerce(),
                    charges: charges.coerce(),
                });

                result[0] +=
                    T::Complex::complex(to::<_, T::Real>(res[0][0]), to::<_, T::Real>(res[0][1]))
                        .mul_real(m_inv_4pi);

                result[1] +=
                    T::Complex::complex(to::<_, T::Real>(res[1][0]), to::<_, T::Real>(res[1][1]))
                        .mul_real(m_inv_4pi);

                result[2] +=
                    T::Complex::complex(to::<_, T::Real>(res[2][0]), to::<_, T::Real>(res[2][1]))
                        .mul_real(m_inv_4pi);

                result[3] +=
                    T::Complex::complex(to::<_, T::Real>(res[3][0]), to::<_, T::Real>(res[3][1]))
                        .mul_real(m_inv_4pi);
            } else {
                panic!()
            }
        }
    }
}

/// Assemble Helmholtz kernel for one target
pub fn assemble_helmholtz_one_target<T: RlstScalar<Complex = T>>(
    eval_type: EvalType,
    target: &[<T as RlstScalar>::Real],
    sources: &[<T as RlstScalar>::Real],
    wavenumber: T::Real,
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

                let kr = wavenumber * diff_norm;

                let g_re = <T::Real as RlstScalar>::cos(kr) * inv_diff_norm * m_inv_4pi;
                let g_im = <T::Real as RlstScalar>::sin(kr) * inv_diff_norm * m_inv_4pi;

                result[index] = <T as RlstScalar>::complex(g_re, g_im);
            }
        }
        EvalType::ValueDeriv => {
            // Cannot simply use an array my_result as this is not
            // correctly auto-vectorized.

            let mut my_result0;
            let mut my_result1;
            let mut my_result2;
            let mut my_result3;

            let mut my_result1_real: T::Real;
            let mut my_result2_real: T::Real;
            let mut my_result3_real: T::Real;

            let mut my_result1_imag: T::Real;
            let mut my_result2_imag: T::Real;
            let mut my_result3_imag: T::Real;

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
                let inv_diff_norm_squared = inv_diff_norm * inv_diff_norm;

                let kr = wavenumber * diff_norm;
                let g_re = <T::Real as RlstScalar>::cos(kr) * inv_diff_norm * m_inv_4pi;
                let g_im = <T::Real as RlstScalar>::sin(kr) * inv_diff_norm * m_inv_4pi;

                let g_deriv_im = (g_im - g_re * kr) * inv_diff_norm_squared;
                let g_deriv_re = (g_re + g_im * kr) * inv_diff_norm_squared;

                my_result1_real = g_deriv_re * diff0;
                my_result1_imag = g_deriv_im * diff0;

                my_result2_real = g_deriv_re * diff1;
                my_result2_imag = g_deriv_im * diff1;

                my_result3_real = g_deriv_re * diff2;
                my_result3_imag = g_deriv_im * diff2;

                my_result0 = <T as RlstScalar>::complex(g_re, g_im);
                my_result1 = <T as RlstScalar>::complex(my_result1_real, my_result1_imag);
                my_result2 = <T as RlstScalar>::complex(my_result2_real, my_result2_imag);
                my_result3 = <T as RlstScalar>::complex(my_result3_real, my_result3_imag);

                my_res0[index] = my_result0;
                my_res1[index] = my_result1;
                my_res2[index] = my_result2;
                my_res3[index] = my_result3;
            }
        }
    }
}

fn helmholtz_component_count(eval_type: EvalType) -> usize {
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
    use rlst::prelude::*;

    #[test]
    fn test_helmholtz_3d_f32() {
        let eps = 1E-5;

        let wavenumber: f32 = 1.5;

        let nsources = 19;
        let ntargets = 7;

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let mut sources = rlst_dynamic_array2!(f32, [3, nsources]);
        let mut targets = rlst_dynamic_array2!(f32, [3, ntargets]);
        let mut charges = rlst_dynamic_array1!(c32, [nsources]);
        let mut green_value = rlst_dynamic_array1!(c32, [ntargets]);

        sources.fill_from_equally_distributed(&mut rng);
        targets.fill_from_equally_distributed(&mut rng);
        charges.fill_from_equally_distributed(&mut rng);

        Helmholtz3dKernel::<c32>::new(wavenumber).evaluate_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            charges.data(),
            green_value.data_mut(),
        );

        let mut expected_val = rlst_dynamic_array1!(c32, [ntargets]);
        let mut expected_deriv = rlst_dynamic_array2!(c32, [4, ntargets]);

        for (val, mut deriv, target) in itertools::izip!(
            expected_val.iter_mut(),
            expected_deriv.col_iter_mut(),
            targets.col_iter(),
        ) {
            for (charge, source) in itertools::izip!(charges.iter(), sources.col_iter_mut()) {
                let mut res: [c32; 1] = [c32::from_real(0.0)];
                let mut res_deriv: [c32; 4] = [
                    c32::from_real(0.0),
                    c32::from_real(0.0),
                    c32::from_real(0.0),
                    c32::from_real(0.0),
                ];
                Helmholtz3dKernel::new(wavenumber).greens_fct(
                    EvalType::Value,
                    source.data(),
                    target.data(),
                    res.as_mut_slice(),
                );
                *val += charge * res[0];

                Helmholtz3dKernel::new(wavenumber).greens_fct(
                    EvalType::ValueDeriv,
                    source.data(),
                    target.data(),
                    res_deriv.as_mut_slice(),
                );

                deriv[[0]] += charge * res_deriv[0];
                deriv[[1]] += charge * res_deriv[1];
                deriv[[2]] += charge * res_deriv[2];
                deriv[[3]] += charge * res_deriv[3];
            }
        }

        for target_index in 0..ntargets {
            assert_relative_eq!(
                green_value[[target_index]],
                expected_val[[target_index]],
                epsilon = eps
            );
        }

        let mut actual = rlst::rlst_dynamic_array2!(c32, [4, ntargets]);

        Helmholtz3dKernel::<c32>::new(wavenumber).evaluate_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            charges.data(),
            actual.data_mut(),
        );

        for target_index in 0..ntargets {
            // let mut my_result = [
            //     c32::from_real(0.0),
            //     c32::from_real(0.0),
            //     c32::from_real(0.0),
            //     c32::from_real(0.0),
            // ];
            // Helmholtz3dKernel::<c32>::new(wavenumber).greens_fct(
            //     EvalType::ValueDeriv,
            //     sources.data(),
            //     targets.data(),
            //     my_result.as_mut_slice(),
            // );

            // println!(
            //     "Values: {} {} {}",
            //     my_result[0] * charges[[0]],
            //     green_value[[target_index]],
            //     actual[[0, target_index]]
            // );
            assert_relative_eq!(
                expected_deriv[[0, target_index]],
                actual[[0, target_index]],
                epsilon = 1E-5
            );

            // println!(
            //     "Second deriv: {} {} {}",
            //     my_result[2] * charges[[0]],
            //     expected_deriv[[2, target_index]],
            //     actual[[2, target_index]]
            // );

            assert_relative_eq!(
                expected_deriv[[1, target_index]],
                actual[[1, target_index]],
                epsilon = 1E-5
            );
            assert_relative_eq!(
                expected_deriv[[2, target_index]],
                actual[[2, target_index]],
                epsilon = 1E-5
            );

            assert_relative_eq!(
                expected_deriv[[3, target_index]],
                actual[[3, target_index]],
                epsilon = 1E-5
            );
        }
    }

    // #[test]
    // fn test_assemble_helmholtz_3d() {
    //     let nsources = 3;
    //     let ntargets = 5;
    //     let wavenumber: f64 = 1.5;

    //     let mut sources = rlst::rlst_dynamic_array2!(f64, [nsources, 3]);
    //     let mut targets = rlst::rlst_dynamic_array2!(f64, [ntargets, 3]);

    //     sources.fill_from_seed_equally_distributed(1);
    //     targets.fill_from_seed_equally_distributed(2);

    //     let mut green_value_t = rlst::rlst_dynamic_array2!(c64, [nsources, ntargets]);

    //     Helmholtz3dKernel::<c64>::new(wavenumber).assemble_st(
    //         EvalType::Value,
    //         sources.data(),
    //         targets.data(),
    //         green_value_t.data_mut(),
    //     );

    //     // The matrix needs to be transposed so that the first row corresponds to the first target,
    //     // second row to the second target and so on.

    //     let mut green_value = rlst::rlst_dynamic_array2!(c64, [ntargets, nsources]);
    //     green_value.fill_from(green_value_t.transpose());

    //     for charge_index in 0..nsources {
    //         let mut charges = rlst::rlst_dynamic_array1![c64, [nsources]];
    //         let mut expected = rlst::rlst_dynamic_array1![c64, [ntargets]];
    //         charges[[charge_index]] = c64::complex(1.0, 0.0);

    //         Helmholtz3dKernel::<c64>::new(wavenumber).evaluate_st(
    //             EvalType::Value,
    //             sources.data(),
    //             targets.data(),
    //             charges.data(),
    //             expected.data_mut(),
    //         );

    //         for target_index in 0..ntargets {
    //             assert_relative_eq!(
    //                 green_value[[target_index, charge_index]],
    //                 expected[[target_index]],
    //                 epsilon = 1E-12
    //             );
    //         }
    //     }

    //     let mut green_value_deriv_t = rlst::rlst_dynamic_array2!(c64, [nsources, 4 * ntargets]);

    //     Helmholtz3dKernel::<c64>::new(wavenumber).assemble_st(
    //         EvalType::ValueDeriv,
    //         sources.data(),
    //         targets.data(),
    //         green_value_deriv_t.data_mut(),
    //     );

    //     // The matrix needs to be transposed so that the first row corresponds to the first target, etc.

    //     let mut green_value_deriv = rlst::rlst_dynamic_array2!(c64, [4 * ntargets, nsources]);
    //     green_value_deriv.fill_from(green_value_deriv_t.transpose());

    //     for charge_index in 0..nsources {
    //         let mut charges = rlst::rlst_dynamic_array1![c64, [nsources]];
    //         let mut expected = rlst::rlst_dynamic_array2!(c64, [4, ntargets]);

    //         charges[[charge_index]] = c64::complex(1.0, 0.0);

    //         Helmholtz3dKernel::<c64>::new(wavenumber).evaluate_st(
    //             EvalType::ValueDeriv,
    //             sources.data(),
    //             targets.data(),
    //             charges.data(),
    //             expected.data_mut(),
    //         );

    //         for deriv_index in 0..4 {
    //             for target_index in 0..ntargets {
    //                 assert_relative_eq!(
    //                     green_value_deriv[[4 * target_index + deriv_index, charge_index]],
    //                     expected[[deriv_index, target_index]],
    //                     epsilon = 1E-12
    //                 );
    //             }
    //         }
    //     }
    // }

    // #[test]
    // fn test_assemble_diag_helmholtz_3d() {
    //     let nsources = 5;
    //     let ntargets = 5;
    //     let wavenumber: f64 = 1.5;

    //     let mut sources = rlst::rlst_dynamic_array2!(f64, [nsources, 3]);
    //     let mut targets = rlst::rlst_dynamic_array2!(f64, [ntargets, 3]);

    //     sources.fill_from_seed_equally_distributed(1);
    //     targets.fill_from_seed_equally_distributed(2);

    //     let mut green_value_diag = rlst::rlst_dynamic_array1!(c64, [ntargets]);
    //     let mut green_value_diag_deriv = rlst::rlst_dynamic_array2!(c64, [4, ntargets]);

    //     Helmholtz3dKernel::<c64>::new(wavenumber).assemble_diagonal_st(
    //         EvalType::Value,
    //         sources.data(),
    //         targets.data(),
    //         green_value_diag.data_mut(),
    //     );
    //     Helmholtz3dKernel::<c64>::new(wavenumber).assemble_diagonal_st(
    //         EvalType::ValueDeriv,
    //         sources.data(),
    //         targets.data(),
    //         green_value_diag_deriv.data_mut(),
    //     );

    //     let mut green_value_t = rlst::rlst_dynamic_array2!(c64, [nsources, ntargets]);

    //     Helmholtz3dKernel::<c64>::new(wavenumber).assemble_st(
    //         EvalType::Value,
    //         sources.data(),
    //         targets.data(),
    //         green_value_t.data_mut(),
    //     );

    //     // The matrix needs to be transposed so that the first row corresponds to the first target,
    //     // second row to the second target and so on.

    //     let mut green_value = rlst::rlst_dynamic_array2!(c64, [ntargets, nsources]);
    //     green_value.fill_from(green_value_t.transpose());

    //     let mut green_value_deriv_t = rlst::rlst_dynamic_array2!(c64, [nsources, 4 * ntargets]);

    //     Helmholtz3dKernel::<c64>::new(wavenumber).assemble_st(
    //         EvalType::ValueDeriv,
    //         sources.data(),
    //         targets.data(),
    //         green_value_deriv_t.data_mut(),
    //     );

    //     // The matrix needs to be transposed so that the first row corresponds to the first target, etc.

    //     let mut green_value_deriv = rlst::rlst_dynamic_array2!(c64, [4 * ntargets, nsources]);
    //     green_value_deriv.fill_from(green_value_deriv_t.transpose());

    //     for index in 0..nsources {
    //         assert_relative_eq!(
    //             green_value[[index, index]],
    //             green_value_diag[[index]],
    //             epsilon = 1E-12
    //         );

    //         assert_relative_eq!(
    //             green_value_deriv[[4 * index, index]],
    //             green_value_diag_deriv[[0, index]],
    //             epsilon = 1E-12,
    //         );

    //         assert_relative_eq!(
    //             green_value_deriv[[4 * index + 1, index]],
    //             green_value_diag_deriv[[1, index]],
    //             epsilon = 1E-12,
    //         );

    //         assert_relative_eq!(
    //             green_value_deriv[[4 * index + 2, index]],
    //             green_value_diag_deriv[[2, index]],
    //             epsilon = 1E-12,
    //         );

    //         assert_relative_eq!(
    //             green_value_deriv[[4 * index + 3, index]],
    //             green_value_diag_deriv[[3, index]],
    //             epsilon = 1E-12,
    //         );
    //     }
    // }
}
