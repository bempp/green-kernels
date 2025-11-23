//! Distributed evaluation of sources and targets.

use green_kernels::{
    laplace_3d::Laplace3dKernel,
    traits::{DistributedKernelEvaluator, Kernel},
    types::GreenKernelEvalType,
};
use mpi::traits::{Communicator, Root};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rlst::{assert_array_relative_eq, rlst_dynamic_array};

fn main() {
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

    let mut sources = rlst_dynamic_array!(f64, [3 * n_sources]);
    let mut targets = rlst_dynamic_array!(f64, [3 * n_targets]);
    let mut charges = rlst_dynamic_array!(f64, [n_sources]);

    sources.fill_from_equally_distributed(&mut rng);
    targets.fill_from_equally_distributed(&mut rng);
    charges.fill_from_equally_distributed(&mut rng);

    // Create the result vector.
    let mut result = rlst_dynamic_array!(f64, [n_targets]);

    kernel.evaluate_distributed(
        GreenKernelEvalType::Value,
        sources.data().unwrap(),
        targets.data().unwrap(),
        charges.data().unwrap(),
        result.data_mut().unwrap(),
        false,
        &world,
    );

    // We now check the result with an evaluation only on the first rank.

    if world.rank() != 0 {
        let root_process = world.process_at_rank(0);

        root_process.gather_into(sources.data().unwrap());
        root_process.gather_into(targets.data().unwrap());
        root_process.gather_into(charges.data().unwrap());
        root_process.gather_into(result.data().unwrap());
    } else {
        let sources = {
            let mut tmp = rlst_dynamic_array!(f64, [3 * n_sources * world.size() as usize]);
            world
                .this_process()
                .gather_into_root(sources.data().unwrap(), tmp.data_mut().unwrap());
            tmp
        };

        let targets = {
            let mut tmp = rlst_dynamic_array!(f64, [3 * n_targets * world.size() as usize]);
            world
                .this_process()
                .gather_into_root(targets.data().unwrap(), tmp.data_mut().unwrap());
            tmp
        };

        let charges = {
            let mut tmp = rlst_dynamic_array!(f64, [n_sources * world.size() as usize]);
            world
                .this_process()
                .gather_into_root(charges.data().unwrap(), tmp.data_mut().unwrap());
            tmp
        };

        let result = {
            let mut tmp = rlst_dynamic_array!(f64, [n_targets * world.size() as usize]);
            world
                .this_process()
                .gather_into_root(result.data().unwrap(), tmp.data_mut().unwrap());
            tmp
        };

        let mut expected = rlst_dynamic_array!(f64, [n_targets * world.size() as usize]);

        kernel.evaluate_mt(
            GreenKernelEvalType::Value,
            sources.data().unwrap(),
            targets.data().unwrap(),
            charges.data().unwrap(),
            expected.data_mut().unwrap(),
        );

        assert_array_relative_eq!(result, expected, 1e-13);
    }
}
