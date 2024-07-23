//! C Interface

use paste::paste;
use rlst::prelude::*;
use rlst::RlstScalar;
use std::{ffi::c_void, mem::ManuallyDrop};

use crate::{laplace_3d::Laplace3dKernel, traits::Kernel, types::EvalType};

pub struct KernelEvaluator<T: RlstScalar> {
    kernel: Box<dyn Kernel<T = T>>,
}

unsafe fn evaluator_from_ptr<T: RlstScalar>(
    kernel_p: *mut c_void,
) -> ManuallyDrop<Box<KernelEvaluator<T>>> {
    unsafe { ManuallyDrop::new(Box::from_raw(kernel_p as *mut KernelEvaluator<T>)) }
}

// Constructors

macro_rules! laplace {
    ($scalar:ty) => {
        paste! {
            #[no_mangle]
            pub extern "C" fn [<create_laplace_3d_evaluator_ $scalar>]() -> *mut c_void {
                let evaluator = Box::new(KernelEvaluator {
                    kernel: Box::new(Laplace3dKernel::<f32>::new()),
                });

                Box::into_raw(evaluator) as *mut std::ffi::c_void
            }
        }
    };
}

laplace!(f32);
laplace!(f64);

// Trait methods

macro_rules! impl_evaluator {
    ($scalar:ty) => {
        paste! {
        #[no_mangle]
        pub extern "C" fn [<destroy_evaluator_ $scalar>](kernel_p: *mut c_void) {
            unsafe {
                ManuallyDrop::into_inner(evaluator_from_ptr::<$scalar>(kernel_p));
            };
        }

        #[no_mangle]
        pub extern "C" fn [<evaluate_st_ $scalar>](
            eval_type: EvalType,
            kernel_p: *mut c_void,
            nsources: usize,
            ntargets: usize,
            sources: *const <$scalar as RlstScalar>::Real,
            targets: *const <$scalar as RlstScalar>::Real,
            charges: *const $scalar,
            result: *mut $scalar,
            dim: usize,
            multithreaded: bool,
        ) {
            let evaluator = unsafe { evaluator_from_ptr::<$scalar>(kernel_p) };
            let sources = unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
            let targets = unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
            let charges = unsafe { std::slice::from_raw_parts(charges, nsources) };
            let result = unsafe { std::slice::from_raw_parts_mut(result, ntargets) };

            if multithreaded {
                evaluator
                    .kernel
                    .evaluate_mt(eval_type, sources, targets, charges, result);
            } else {
                evaluator
                    .kernel
                    .evaluate_st(eval_type, sources, targets, charges, result);
            }
        }
        }
    };
}

impl_evaluator!(f64);
impl_evaluator!(f32);
impl_evaluator!(c32);
impl_evaluator!(c64);
