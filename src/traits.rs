//! Trait for Green's function kernels

use crate::types::GreenKernelEvalType;
#[cfg(feature = "mpi")]
use mpi::traits::{Communicator, Equivalence, Root};
use rlst::RlstScalar;
#[cfg(feature = "mpi")]
use rlst::{rlst_dynamic_array1, RawAccess, RawAccessMut};

/// Interface to evaluating Green's functions for given sources and targets.
pub trait Kernel: Sync {
    /// The scalar type
    type T: RlstScalar;

    /// Evaluate the Green's fct. for a single source and single target.
    fn greens_fct(
        &self,
        eval_type: GreenKernelEvalType,
        source: &[<Self::T as RlstScalar>::Real],
        target: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    );

    /// Single threaded evaluation of Green's functions.
    ///
    /// - `eval_type`: Either [EvalType::Value] to only return Green's function values
    ///              or [EvalType::ValueDeriv] to return values and derivatives.
    /// - `sources`: A slice defining the source points. The points must be given in the form
    ///            `[x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N]`, that is
    ///            the value for each dimension must be continuously contained in the slice.
    /// - `targets`: A slice defining the targets. The memory layout is the same as for sources.
    /// - `charges`: A slice defining the charges. For each source point there needs to be one charge.
    /// - `result`: The result array. If the kernel is RlstScalar and `eval_type` has the value [EvalType::Value]
    ///           then `result` has the same number of elemens as there are targets. For a RlstScalar kernel
    ///           in three dimensional space if [EvalType::ValueDeriv] was chosen then `result` contains
    ///           for each target in consecutive order the value of the kernel and the three components
    ///           of its derivative.
    ///
    fn evaluate_st(
        &self,
        eval_type: GreenKernelEvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
    );

    /// Multi-threaded evaluation of a Green's function kernel.
    ///
    /// The method parallelizes over the given targets. It expects a Rayon `ThreadPool`
    /// in which the multi-threaded execution can be scheduled.
    fn evaluate_mt(
        &self,
        eval_type: GreenKernelEvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
    );

    /// Single threaded assembly of a kernel matrix.
    ///
    /// - `eval_type`: Either [EvalType::Value] to only return Green's function values
    ///              or [EvalType::ValueDeriv] to return values and derivatives.
    /// - `sources`: A slice defining the source points. The points must be given in the form
    ///            `[x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N]`, that is
    ///            the value for each dimension must be continuously contained in the slice.
    /// - `targets`: A slice defining the targets. The memory layout is the same as for sources.
    /// - `result`: The result array. If the kernel is RlstScalar and `eval_type` has the value [EvalType::Value]
    ///           then `result` is equivalent to a column major matrix of dimension [S, T], where S is the number of sources and
    ///           T is the number of targets. Hence, for each target all corresponding source evaluations are consecutively in memory.
    ///           For a RlstScalar kernel in three dimensional space if [EvalType::ValueDeriv] was chosen then `result` is equivalent
    ///           to a column-major matrix of dimension [4 * S, T], where the first 4 rows are the values of Green's fct. value and
    ///           derivatives for the first source and all targets. The next 4 rows correspond to values and derivatives of second source
    ///           with all targets and so on.
    ///
    fn assemble_st(
        &self,
        eval_type: GreenKernelEvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    );

    /// Multi-threaded version of kernel matrix assembly.
    fn assemble_mt(
        &self,
        eval_type: GreenKernelEvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    );

    /// Single threaded assembly of the diagonal of a kernel matrix
    fn assemble_pairwise_st(
        &self,
        eval_type: GreenKernelEvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    );

    /// Return the domain component count of the Green's fct.
    ///
    /// For a RlstScalar kernel this is `1`.
    fn domain_component_count(&self) -> usize;

    /// Return the space dimension.
    fn space_dimension(&self) -> usize;

    /// Return the range component count of the Green's fct.
    ///
    /// For a RlstScalar kernel this is `1` if [EvalType::Value] is
    /// given, and `4` if [EvalType::ValueDeriv] is given.
    fn range_component_count(&self, eval_type: GreenKernelEvalType) -> usize;
}

// Note that we cannot just add the `evaluate_distributed` method to the `Kernel` trait
// since currently the C interface is implemented by making `Kernel` a trait object.
// This requires that methods do not introduce additional template parameters. Can change this
// again once we move to the better C interface in `c-api-tools`.

/// Distributed evaluation of a Green's function kernel.
///
/// If `use_multithreaded` is set to true, the evaluation uses Rayon multi-threading on each rank.
/// Otherwise, the evaluation on each rank is single-threaded.
#[cfg(feature = "mpi")]
pub trait DistributedKernelEvaluator: Kernel {
    fn evaluate_distributed<C: Communicator>(
        &self,
        eval_type: GreenKernelEvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
        use_multithreaded: bool,
        comm: &C,
    ) where
        Self::T: Equivalence,
        <Self::T as RlstScalar>::Real: Equivalence,
    {
        // Check that the number of sources and number of charges are compatible.
        assert_eq!(sources.len(), 3 * charges.len());

        // Check that the output vector has the correct size.
        // Multiply result by 3 since targets have 3 components (x, y, z) direction.
        assert_eq!(
            self.range_component_count(eval_type) * targets.len(),
            3 * result.len()
        );

        let size = comm.size();

        // We now iterate through each rank associated with the sources and communicate from that rank
        // the sources to all target ranks.

        for rank in 0..size as usize {
            // Communicate the sources and charges from `rank` to all ranks.

            // We first need to tell all ranks how many sources and charges we have.
            let root_process = comm.process_at_rank(rank as i32);

            let nsources = {
                let mut nsources;
                if comm.rank() == rank as i32 {
                    nsources = charges.len();
                } else {
                    nsources = 0;
                }
                root_process.broadcast_into(&mut nsources);
                nsources
            };

            let mut root_sources =
                rlst_dynamic_array1!(<Self::T as RlstScalar>::Real, [3 * nsources]);
            let mut root_charges = rlst_dynamic_array1!(Self::T, [nsources]);

            if comm.rank() == rank as i32 {
                root_sources.data_mut().copy_from_slice(sources);
                root_charges.data_mut().copy_from_slice(charges);
            }

            root_process.broadcast_into(&mut root_sources.data_mut()[..]);
            root_process.broadcast_into(&mut root_charges.data_mut()[..]);

            // We now have the sources and charges on all ranks. We can now simply evaluate.

            if use_multithreaded {
                self.evaluate_mt(
                    eval_type,
                    &root_sources.data()[..],
                    targets,
                    &root_charges.data()[..],
                    result,
                );
            } else {
                self.evaluate_st(
                    eval_type,
                    &root_sources.data()[..],
                    targets,
                    &root_charges.data()[..],
                    result,
                );
            }
        }
    }
}

#[cfg(feature = "mpi")]
impl<K: Kernel> DistributedKernelEvaluator for K {}
