//! Trait for Green's function kernels

use crate::types::GreenKernelEvalType;
#[cfg(feature = "mpi")]
use mpi::traits::{Communicator, Equivalence, Root};
use rlst::RlstScalar;
#[cfg(feature = "mpi")]
use rlst::{rlst_dynamic_array1, DistributedVector, IndexLayout, RawAccess, RawAccessMut};

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
#[cfg(feature = "mpi")]
pub trait DistributedKernelEvaluator: Kernel {
    fn evaluate_distributed<
        SourceLayout: IndexLayout,
        TargetLayout: IndexLayout,
        ChargeLayout: IndexLayout,
        ResultLayout: IndexLayout,
    >(
        &self,
        eval_type: GreenKernelEvalType,
        sources: &DistributedVector<'_, SourceLayout, <Self::T as RlstScalar>::Real>,
        targets: &DistributedVector<'_, TargetLayout, <Self::T as RlstScalar>::Real>,
        charges: &DistributedVector<'_, ChargeLayout, Self::T>,
        result: &mut DistributedVector<'_, ResultLayout, Self::T>,
    ) where
        Self::T: Equivalence,
        <Self::T as RlstScalar>::Real: Equivalence,
    {
        // We want that everything has the same communicator
        assert!(std::ptr::addr_eq(
            sources.index_layout().comm(),
            charges.index_layout().comm()
        ));
        assert!(std::ptr::addr_eq(
            sources.index_layout().comm(),
            targets.index_layout().comm()
        ));
        assert!(std::ptr::addr_eq(
            sources.index_layout().comm(),
            result.index_layout().comm()
        ));

        // Check that the output vector has the correct size.
        assert_eq!(
            targets.index_layout().number_of_local_indices(),
            3 * result.index_layout().number_of_local_indices()
        );

        let size = sources.index_layout().comm().size();

        // We now iterate through each rank associated with the sources and communicate from that rank
        // the sources to all target ranks.

        for rank in 0..size as usize {
            // Communicate the sources and charges from `rank` to all ranks.

            let root_process = sources.index_layout().comm().process_at_rank(rank as i32);
            let source_range = sources.index_layout().index_range(rank).unwrap();
            let charge_range = charges.index_layout().index_range(rank).unwrap();
            let nsources = source_range.1 - source_range.0;
            let ncharges = charge_range.1 - charge_range.0;
            // Make sure that number of sources and charges are compatible.
            assert_eq!(nsources, 3 * ncharges);
            let mut root_sources = rlst_dynamic_array1!(<Self::T as RlstScalar>::Real, [nsources]);
            let mut root_charges = rlst_dynamic_array1!(Self::T, [ncharges]);
            // If we are on `rank` fill the sources and charges.
            if sources.index_layout().comm().rank() == rank as i32 {
                root_sources.fill_from(sources.local().r());
                root_charges.fill_from(charges.local().r());
            }

            root_process.broadcast_into(&mut root_sources.data_mut()[..]);
            root_process.broadcast_into(&mut root_charges.data_mut()[..]);

            // We now have the sources and charges on all ranks. We can now simply evaluate.
            self.evaluate_mt(
                eval_type,
                &root_sources.data()[..],
                targets.local().data(),
                &root_charges.data()[..],
                result.local_mut().data_mut(),
            );
        }
    }
}

#[cfg(feature = "mpi")]
impl<K: Kernel> DistributedKernelEvaluator for K {}
