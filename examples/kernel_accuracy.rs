//! Test the accuracy of the inverse sqrt

const NSAMPLES: usize = 100;
use num::{traits::FloatConst, Float};
use rand::prelude::*;
use rlst::{dense::tools::RandScalar, prelude::*};

use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel,
    modified_helmholtz_3d::ModifiedHelmholtz3dKernel, traits::Kernel, types::EvalType,
};

fn benchmark_kernel_laplace<T: RlstScalar + RandScalar, K: Kernel<T = T>>(
    kernel: &K,
    sources: &DynamicArray<T::Real, 2>,
    targets: &DynamicArray<T::Real, 2>,
) -> T::Real
where
    T::Real: num::Float,
{
    let mut result = rlst_dynamic_array2!(T, [NSAMPLES, NSAMPLES]);

    kernel.assemble_mt(
        EvalType::Value,
        sources.data(),
        targets.data(),
        result.data_mut(),
    );

    let mut rel_error: T::Real = <T::Real>::default();

    for (source_index, source) in sources.col_iter().enumerate() {
        for (target_index, target) in targets.col_iter().enumerate() {
            let diff_norm = (source.view() - target.view()).norm_2();
            let green = result[[source_index, target_index]];
            let green_exact = T::one()
                / (num::cast::<f64, T>(4.0 * f64::PI()).unwrap()
                    * num::cast::<T::Real, T>(diff_norm).unwrap());

            rel_error =
                rel_error.max((green - green_exact).abs() / green.abs().max(green_exact.abs()));
        }
    }

    rel_error
}

fn benchmark_kernel_modified_helmholtz<T: RlstScalar + RandScalar, K: Kernel<T = T>>(
    kernel: &K,
    sources: &DynamicArray<T::Real, 2>,
    targets: &DynamicArray<T::Real, 2>,
) -> T::Real
where
    T::Real: num::Float,
{
    let mut result = rlst_dynamic_array2!(T, [NSAMPLES, NSAMPLES]);

    kernel.assemble_mt(
        EvalType::Value,
        sources.data(),
        targets.data(),
        result.data_mut(),
    );

    let mut rel_error: T::Real = <T::Real>::default();

    for (source_index, source) in sources.col_iter().enumerate() {
        for (target_index, target) in targets.col_iter().enumerate() {
            let diff_norm = (source.view() - target.view()).norm_2();
            let green = result[[source_index, target_index]];
            let green_exact = T::exp(T::from_real(
                num::cast::<f64, T::Real>(-1.5).unwrap() * diff_norm,
            )) / (num::cast::<f64, T>(4.0 * f64::PI()).unwrap()
                * num::cast::<T::Real, T>(diff_norm).unwrap());

            rel_error =
                rel_error.max((green - green_exact).abs() / green.abs().max(green_exact.abs()));
        }
    }

    rel_error
}

fn benchmark_kernel_helmholtz<T: RlstScalar<Complex = T> + RandScalar, K: Kernel<T = T>>(
    kernel: &K,
    sources: &DynamicArray<T::Real, 2>,
    targets: &DynamicArray<T::Real, 2>,
) -> T::Real
where
    T::Real: num::Float,
{
    let mut result = rlst_dynamic_array2!(T, [NSAMPLES, NSAMPLES]);

    kernel.assemble_mt(
        EvalType::Value,
        sources.data(),
        targets.data(),
        result.data_mut(),
    );

    let mut rel_error: T::Real = <T::Real>::default();

    for (source_index, source) in sources.col_iter().enumerate() {
        for (target_index, target) in targets.col_iter().enumerate() {
            let diff_norm = (source.view() - target.view()).norm_2();
            let green = result[[source_index, target_index]];
            let green_exact = T::complex(
                T::cos(T::from_real(
                    num::cast::<f64, T::Real>(1.5).unwrap() * diff_norm,
                )) / (num::cast::<f64, T>(4.0 * f64::PI()).unwrap()
                    * num::cast::<T::Real, T>(diff_norm).unwrap()),
                T::sin(T::from_real(
                    num::cast::<f64, T::Real>(1.5).unwrap() * diff_norm,
                )) / (num::cast::<f64, T>(4.0 * f64::PI()).unwrap()
                    * num::cast::<T::Real, T>(diff_norm).unwrap()),
            );

            rel_error =
                rel_error.max((green - green_exact).abs() / green.abs().max(green_exact.abs()));
        }
    }

    rel_error
}

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let mut sources_f32 = rlst_dynamic_array2!(f32, [3, NSAMPLES]);
    let mut targets_f32 = rlst_dynamic_array2!(f32, [3, NSAMPLES]);
    let mut sources_f64 = rlst_dynamic_array2!(f64, [3, NSAMPLES]);
    let mut targets_f64 = rlst_dynamic_array2!(f64, [3, NSAMPLES]);

    sources_f32.fill_from_equally_distributed(&mut rng);
    targets_f32.fill_from_equally_distributed(&mut rng);
    sources_f64.fill_from_equally_distributed(&mut rng);
    targets_f64.fill_from_equally_distributed(&mut rng);

    let laplace_f32 =
        benchmark_kernel_laplace(&Laplace3dKernel::<f32>::new(), &sources_f32, &targets_f32);
    let laplace_f64 =
        benchmark_kernel_laplace(&Laplace3dKernel::<f64>::new(), &sources_f64, &targets_f64);

    let modified_helmholtz_f32 = benchmark_kernel_modified_helmholtz(
        &ModifiedHelmholtz3dKernel::<f32>::new(1.5),
        &sources_f32,
        &targets_f32,
    );
    let modified_helmholtz_f64 = benchmark_kernel_modified_helmholtz(
        &ModifiedHelmholtz3dKernel::<f64>::new(1.5),
        &sources_f64,
        &targets_f64,
    );

    let helmholtz_f32 = benchmark_kernel_helmholtz(
        &Helmholtz3dKernel::<c32>::new(1.5),
        &sources_f32,
        &targets_f32,
    );
    let helmholtz_f64 = benchmark_kernel_helmholtz(
        &Helmholtz3dKernel::<c64>::new(1.5),
        &sources_f64,
        &targets_f64,
    );

    println!("Laplace maximum relative error: {:.2E}", laplace_f32);
    println!("Laplace maximum relative error: {:.2E}", laplace_f64);

    println!(
        "Modified Helmholtz maximum relative error: {:.2E}",
        modified_helmholtz_f32
    );
    println!(
        "Modified Helmholtz maximum relative error: {:.2E}",
        modified_helmholtz_f64
    );

    println!("Helmholtz maximum relative error: {:.2E}", helmholtz_f32);
    println!("Helmholtz maximum relative error: {:.2E}", helmholtz_f64);
}
