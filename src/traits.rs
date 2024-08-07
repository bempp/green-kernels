//! Trait for Green's function kernels
use crate::types::GreenKernelEvalType;
use rlst::RlstScalar;

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
