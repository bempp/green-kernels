//! Distributed evaluation of sources and targets.

use bempp_distributed_tools::IndexLayoutFromLocalCounts;
use green_kernels::traits::*;
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use mpi::traits::Communicator;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rlst::prelude::*;
use rlst::{
    assert_array_relative_eq, rlst_dynamic_array1, DistributedVector, RawAccess, RawAccessMut,
};

fn main() {
    // Ensure that there is only one Rayon thread per process

    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    // Create the MPI communicator
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    // Number of sources on each process.
    let n_sources = 10000;
    // Number of targets on each process.
    let n_targets = 1000;

    // Init the random number generator.
    let mut rng = ChaCha8Rng::seed_from_u64(0);

    // Create a Laplace kernel.
    let kernel = Laplace3dKernel::<f64>::default();

    // We create index layout for sources and targets.
    let source_layout = IndexLayoutFromLocalCounts::new(3 * n_sources, &world);
    let target_layout = IndexLayoutFromLocalCounts::new(3 * n_targets, &world);
    let charge_layout = IndexLayoutFromLocalCounts::new(n_sources, &world);
    let result_layout = IndexLayoutFromLocalCounts::new(n_targets, &world);

    // Create the sources and charges.
    let sources = DistributedVector::<_, f64>::new(&source_layout);
    let targets = DistributedVector::<_, f64>::new(&target_layout);

    sources.local_mut().fill_from_equally_distributed(&mut rng);
    targets.local_mut().fill_from_equally_distributed(&mut rng);

    // Create the charges.
    let charges = DistributedVector::<_, f64>::new(&charge_layout);
    charges.local_mut().fill_from_equally_distributed(&mut rng);

    // Create the result vector.
    let mut result = DistributedVector::<_, f64>::new(&result_layout);

    // Evaluate the kernel.

    kernel.evaluate_distributed(
        GreenKernelEvalType::Value,
        &sources,
        &targets,
        &charges,
        &mut result,
    );

    // We now check the result with an evaluation only on the first rank.

    if world.rank() != 0 {
        sources.gather_to_rank(0);
        targets.gather_to_rank(0);
        charges.gather_to_rank(0);
        result.gather_to_rank(0);
    } else {
        let sources = {
            let mut tmp = rlst_dynamic_array1!(f64, [3 * n_sources * world.size() as usize]);
            sources.gather_to_rank_root(tmp.r_mut());
            tmp
        };

        let targets = {
            let mut tmp = rlst_dynamic_array1!(f64, [3 * n_targets * world.size() as usize]);
            targets.gather_to_rank_root(tmp.r_mut());
            tmp
        };

        let charges = {
            let mut tmp = rlst_dynamic_array1!(f64, [n_sources * world.size() as usize]);
            charges.gather_to_rank_root(tmp.r_mut());
            tmp
        };

        let result = {
            let mut tmp = rlst_dynamic_array1!(f64, [n_targets * world.size() as usize]);
            result.gather_to_rank_root(tmp.r_mut());
            tmp
        };

        let mut expected = rlst_dynamic_array1!(f64, [n_targets * world.size() as usize]);

        kernel.evaluate_mt(
            GreenKernelEvalType::Value,
            sources.data(),
            targets.data(),
            charges.data(),
            expected.data_mut(),
        );

        assert_array_relative_eq!(result, expected, 1e-13);
    }
}
