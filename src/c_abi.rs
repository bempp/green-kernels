//! C Interface

use coe;
use rlst::prelude::*;
use rlst::RlstScalar;
use std::{ffi::c_void, mem::ManuallyDrop};

use crate::{laplace_3d::Laplace3dKernel, traits::Kernel, types::GreenKernelEvalType};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum GreenKernelCType {
    F32,
    F64,
    C32,
    C64,
}

pub struct GreenKernelEvaluator {
    ctype: GreenKernelCType,
    kernel_p: *mut c_void,
}

impl GreenKernelEvaluator {
    pub fn get_ctype(&self) -> GreenKernelCType {
        self.ctype
    }

    pub fn get_kernel(&self) -> *mut c_void {
        self.kernel_p
    }
}

impl Drop for GreenKernelEvaluator {
    fn drop(&mut self) {
        let Self { ctype, kernel_p } = self;
        match ctype {
            GreenKernelCType::F32 => {
                drop(unsafe { Box::from_raw(*kernel_p as *mut Box<dyn Kernel<T = f32>>) });
            }
            GreenKernelCType::F64 => {
                drop(unsafe { Box::from_raw(*kernel_p as *mut Box<dyn Kernel<T = f64>>) });
            }
            GreenKernelCType::C32 => {
                drop(unsafe { Box::from_raw(*kernel_p as *mut Box<dyn Kernel<T = c32>>) });
            }
            GreenKernelCType::C64 => {
                drop(unsafe { Box::from_raw(*kernel_p as *mut Box<dyn Kernel<T = c64>>) });
            }
        }
    }
}

fn green_kernel_inner<T: RlstScalar>(
    kernel_p: *mut GreenKernelEvaluator,
) -> ManuallyDrop<Box<Box<dyn Kernel<T = T>>>> {
    assert!(!kernel_p.is_null());
    green_kernel_assert_type::<T>(kernel_p);
    let kernel_p = unsafe { (*kernel_p).kernel_p };
    ManuallyDrop::new(unsafe { Box::from_raw(kernel_p as *mut Box<dyn Kernel<T = T>>) })
}

fn green_kernel_assert_type<T: RlstScalar>(kernel_p: *mut GreenKernelEvaluator) {
    assert!(!kernel_p.is_null());
    let ctype = green_kernel_get_ctype(kernel_p);
    match ctype {
        GreenKernelCType::F32 => coe::assert_same::<f32, T>(),
        GreenKernelCType::F64 => coe::assert_same::<f64, T>(),
        GreenKernelCType::C32 => coe::assert_same::<c32, T>(),
        GreenKernelCType::C64 => coe::assert_same::<c64, T>(),
    }
}

#[no_mangle]
pub extern "C" fn green_kernel_get_ctype(kernel_p: *mut GreenKernelEvaluator) -> GreenKernelCType {
    assert!(!kernel_p.is_null());
    unsafe { (*kernel_p).get_ctype() }
}

#[no_mangle]
pub extern "C" fn green_kernel_free(kernel_p: *mut GreenKernelEvaluator) {
    assert!(!kernel_p.is_null());
    unsafe { drop(Box::from_raw(kernel_p)) }
}

#[no_mangle]
pub extern "C" fn green_kernel_laplace_3d_alloc(
    ctype: GreenKernelCType,
) -> *mut GreenKernelEvaluator {
    match ctype {
        GreenKernelCType::F32 => {
            let evaluator = Box::new(GreenKernelEvaluator {
                ctype,
                kernel_p: Box::into_raw(Box::new(
                    Box::new(Laplace3dKernel::<f32>::new()) as Box<dyn Kernel<T = f32>>
                )) as *mut c_void,
            });
            Box::into_raw(evaluator)
        }
        GreenKernelCType::F64 => {
            let evaluator = Box::new(GreenKernelEvaluator {
                ctype,
                kernel_p: Box::into_raw(Box::new(
                    Box::new(Laplace3dKernel::<f64>::new()) as Box<dyn Kernel<T = f64>>
                )) as *mut c_void,
            });
            Box::into_raw(evaluator)
        }
        _ => panic!("Unknown type!"),
    }
}

#[no_mangle]
pub extern "C" fn green_kernel_evaluate(
    kernel_p: *mut GreenKernelEvaluator,
    eval_type: GreenKernelEvalType,
    nsources: usize,
    ntargets: usize,
    sources: *const c_void,
    targets: *const c_void,
    charges: *const c_void,
    result: *mut c_void,
    dim: usize,
    multithreaded: bool,
) {
    fn impl_evaluate<T: RlstScalar>(
        kernel_p: *mut GreenKernelEvaluator,
        eval_type: GreenKernelEvalType,
        nsources: usize,
        ntargets: usize,
        sources: *const c_void,
        targets: *const c_void,
        charges: *const c_void,
        result: *const c_void,
        dim: usize,
        multithreaded: bool,
    ) {
        green_kernel_assert_type::<T>(kernel_p);
        let kernel = green_kernel_inner::<T>(kernel_p);
        let range_count = kernel.range_component_count(eval_type);
        let sources: &[T::Real] =
            unsafe { std::slice::from_raw_parts(sources as *const T::Real, nsources * dim) };
        let targets: &[T::Real] =
            unsafe { std::slice::from_raw_parts(targets as *const T::Real, ntargets * dim) };
        let charges: &[T] = unsafe { std::slice::from_raw_parts(charges as *const T, nsources) };
        let result: &mut [T] =
            unsafe { std::slice::from_raw_parts_mut(result as *mut T, ntargets * range_count) };
        if multithreaded {
            kernel.evaluate_mt(eval_type, sources, targets, charges, result);
        } else {
            kernel.evaluate_st(eval_type, sources, targets, charges, result);
        }
    }

    assert!(!kernel_p.is_null());

    match green_kernel_get_ctype(kernel_p) {
        GreenKernelCType::F32 => {
            impl_evaluate::<f32>(
                kernel_p,
                eval_type,
                nsources,
                ntargets,
                sources,
                targets,
                charges,
                result,
                dim,
                multithreaded,
            );
        }
        GreenKernelCType::F64 => {
            impl_evaluate::<f64>(
                kernel_p,
                eval_type,
                nsources,
                ntargets,
                sources,
                targets,
                charges,
                result,
                dim,
                multithreaded,
            );
        }
        GreenKernelCType::C32 => {
            impl_evaluate::<c32>(
                kernel_p,
                eval_type,
                nsources,
                ntargets,
                sources,
                targets,
                charges,
                result,
                dim,
                multithreaded,
            );
        }
        GreenKernelCType::C64 => {
            impl_evaluate::<c64>(
                kernel_p,
                eval_type,
                nsources,
                ntargets,
                sources,
                targets,
                charges,
                result,
                dim,
                multithreaded,
            );
        }
    }
}

#[no_mangle]
pub extern "C" fn green_kernel_range_component_count(
    kernel_p: *mut GreenKernelEvaluator,
    eval_type: GreenKernelEvalType,
) -> u32 {
    assert!(!kernel_p.is_null());
    match green_kernel_get_ctype(kernel_p) {
        GreenKernelCType::F32 => {
            green_kernel_inner::<f32>(kernel_p).range_component_count(eval_type) as u32
        }
        GreenKernelCType::F64 => {
            green_kernel_inner::<f64>(kernel_p).range_component_count(eval_type) as u32
        }
        GreenKernelCType::C32 => {
            green_kernel_inner::<c32>(kernel_p).range_component_count(eval_type) as u32
        }
        GreenKernelCType::C64 => {
            green_kernel_inner::<c64>(kernel_p).range_component_count(eval_type) as u32
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_create_laplace_evaluator() {
        let evaluator = green_kernel_laplace_3d_alloc(GreenKernelCType::F32);
        assert_eq!(GreenKernelCType::F32, green_kernel_get_ctype(evaluator));

        green_kernel_free(evaluator);
    }
}
