//! Test the accuracy of the inverse sqrt

const NSAMPLES: usize = 100;
use num::traits::FloatConst;
use rand::prelude::*;
use rlst::prelude::*;

use green_kernels::{traits::Kernel, types::EvalType, *};

fn main() {
    let nsources = NSAMPLES;
    let ntargets = NSAMPLES;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let mut sources = rlst_dynamic_array2!(f32, [3, nsources]);
    let mut targets = rlst_dynamic_array2!(f32, [3, ntargets]);
    let mut result = rlst_dynamic_array2!(f32, [nsources, ntargets]);

    sources.fill_from_standard_normal(&mut rng);
    targets.fill_from_standard_normal(&mut rng);

    laplace_3d::Laplace3dKernel::<f32>::new().assemble_mt(
        EvalType::Value,
        sources.data(),
        targets.data(),
        result.data_mut(),
    );

    let mut rel_error: f32 = 0.0;

    for (source_index, source) in sources.col_iter().enumerate() {
        for (target_index, target) in targets.col_iter().enumerate() {
            let diff_norm = (source.view() - target.view()).norm_2();
            let green = result[[source_index, target_index]];
            let green_exact = 1.0 / (4.0 * f32::PI() * diff_norm);

            rel_error = rel_error.max((green - green_exact).abs() / green.max(green_exact));
        }
    }

    println!("Laplace maximum relative error: {:.2E}", rel_error);
}
