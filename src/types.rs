//! Type definitions

/// Evaluation Mode
#[derive(Clone, Copy)]
#[repr(C)]
pub enum GreenKernelEvalType {
    /// Only values required
    Value,
    /// Both values and derivatives required
    ValueDeriv,
}
