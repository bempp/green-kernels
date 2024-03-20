//! Type definitions

/// Evaluation Mode
#[derive(Clone, Copy)]
pub enum EvalType {
    /// Only values required
    Value,
    /// Both values and derivatives required
    ValueDeriv,
}
