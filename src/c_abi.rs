//! C Interface

use coe;
use rlst::prelude::*;
use rlst::RlstScalar;
use std::{ffi::c_void, mem::ManuallyDrop};

use crate::helmholtz_3d::Helmholtz3dKernel;
use crate::modified_helmholtz_3d::ModifiedHelmholtz3dKernel;
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
    let ctype = unsafe { green_kernel_get_ctype(kernel_p) };
    match ctype {
        GreenKernelCType::F32 => coe::assert_same::<f32, T>(),
        GreenKernelCType::F64 => coe::assert_same::<f64, T>(),
        GreenKernelCType::C32 => coe::assert_same::<c32, T>(),
        GreenKernelCType::C64 => coe::assert_same::<c64, T>(),
    }
}

/// Return the type of the kernel.
///
/// # Safety
/// Pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn green_kernel_get_ctype(
    kernel_p: *mut GreenKernelEvaluator,
) -> GreenKernelCType {
    assert!(!kernel_p.is_null());
    (*kernel_p).get_ctype()
}

/// Free the kernel.
///
/// # Safety
/// Pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn green_kernel_free(kernel_p: *mut GreenKernelEvaluator) {
    assert!(!kernel_p.is_null());
    drop(Box::from_raw(kernel_p))
}

/// Create a new Laplace kernel.
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

/// Create a new Modified Helmholtz kernel.
#[no_mangle]
pub extern "C" fn green_kernel_modified_helmholtz_3d_alloc(
    ctype: GreenKernelCType,
    omega: f64,
) -> *mut GreenKernelEvaluator {
    match ctype {
        GreenKernelCType::F32 => {
            let evaluator = Box::new(GreenKernelEvaluator {
                ctype,
                kernel_p: Box::into_raw(Box::new(Box::new(ModifiedHelmholtz3dKernel::<f32>::new(
                    omega as f32,
                )) as Box<dyn Kernel<T = f32>>)) as *mut c_void,
            });
            Box::into_raw(evaluator)
        }
        GreenKernelCType::F64 => {
            let evaluator = Box::new(GreenKernelEvaluator {
                ctype,
                kernel_p: Box::into_raw(Box::new(Box::new(ModifiedHelmholtz3dKernel::<f64>::new(
                    omega,
                )) as Box<dyn Kernel<T = f64>>)) as *mut c_void,
            });
            Box::into_raw(evaluator)
        }
        _ => panic!("Unknown type!"),
    }
}

/// Create a new Helmholtz kernel.
#[no_mangle]
pub extern "C" fn green_kernel_helmholtz_3d_alloc(
    ctype: GreenKernelCType,
    wavenumber: f64,
) -> *mut GreenKernelEvaluator {
    match ctype {
        GreenKernelCType::C32 => {
            let evaluator = Box::new(GreenKernelEvaluator {
                ctype,
                kernel_p: Box::into_raw(Box::new(Box::new(Helmholtz3dKernel::<c32>::new(
                    num::cast(wavenumber).unwrap(),
                )) as Box<dyn Kernel<T = c32>>)) as *mut c_void,
            });
            Box::into_raw(evaluator)
        }
        GreenKernelCType::C64 => {
            let evaluator = Box::new(GreenKernelEvaluator {
                ctype,
                kernel_p: Box::into_raw(Box::new(Box::new(Helmholtz3dKernel::<c64>::new(
                    num::cast(wavenumber).unwrap(),
                )) as Box<dyn Kernel<T = c64>>)) as *mut c_void,
            });
            Box::into_raw(evaluator)
        }
        _ => panic!("Unknown type!"),
    }
}

/// Evaluate a kernel.
///
/// # Safety
/// Pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn green_kernel_evaluate(
    kernel_p: *mut GreenKernelEvaluator,
    eval_type: GreenKernelEvalType,
    nsources: usize,
    ntargets: usize,
    sources: *const c_void,
    targets: *const c_void,
    charges: *const c_void,
    result: *mut c_void,
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
        multithreaded: bool,
    ) {
        let kernel = green_kernel_inner::<T>(kernel_p);
        let range_count = kernel.range_component_count(eval_type);
        let dim = unsafe { green_kernel_space_dimension(kernel_p) as usize };
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

    match unsafe { green_kernel_get_ctype(kernel_p) } {
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
                multithreaded,
            );
        }
    }
}

/// Assemble a kernel.
///
/// # Safety
/// Pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn green_kernel_assemble(
    kernel_p: *mut GreenKernelEvaluator,
    eval_type: GreenKernelEvalType,
    nsources: usize,
    ntargets: usize,
    sources: *const c_void,
    targets: *const c_void,
    result: *mut c_void,
    multithreaded: bool,
) {
    fn impl_assemble<T: RlstScalar>(
        kernel_p: *mut GreenKernelEvaluator,
        eval_type: GreenKernelEvalType,
        nsources: usize,
        ntargets: usize,
        sources: *const c_void,
        targets: *const c_void,
        result: *const c_void,
        multithreaded: bool,
    ) {
        let kernel = green_kernel_inner::<T>(kernel_p);
        let range_count = kernel.range_component_count(eval_type);
        let dim = unsafe { green_kernel_space_dimension(kernel_p) as usize };
        let sources: &[T::Real] =
            unsafe { std::slice::from_raw_parts(sources as *const T::Real, nsources * dim) };
        let targets: &[T::Real] =
            unsafe { std::slice::from_raw_parts(targets as *const T::Real, ntargets * dim) };
        let result: &mut [T] = unsafe {
            std::slice::from_raw_parts_mut(result as *mut T, nsources * ntargets * range_count)
        };
        if multithreaded {
            kernel.assemble_mt(eval_type, sources, targets, result);
        } else {
            kernel.assemble_st(eval_type, sources, targets, result);
        }
    }

    assert!(!kernel_p.is_null());

    match green_kernel_get_ctype(kernel_p) {
        GreenKernelCType::F32 => {
            impl_assemble::<f32>(
                kernel_p,
                eval_type,
                nsources,
                ntargets,
                sources,
                targets,
                result,
                multithreaded,
            );
        }
        GreenKernelCType::F64 => {
            impl_assemble::<f64>(
                kernel_p,
                eval_type,
                nsources,
                ntargets,
                sources,
                targets,
                result,
                multithreaded,
            );
        }
        GreenKernelCType::C32 => {
            impl_assemble::<c32>(
                kernel_p,
                eval_type,
                nsources,
                ntargets,
                sources,
                targets,
                result,
                multithreaded,
            );
        }
        GreenKernelCType::C64 => {
            impl_assemble::<c64>(
                kernel_p,
                eval_type,
                nsources,
                ntargets,
                sources,
                targets,
                result,
                multithreaded,
            );
        }
    }
}

/// Pairwise assembly of sources and targets.
///
/// # Safety
/// Pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn green_kernel_pairwise_assemble(
    kernel_p: *mut GreenKernelEvaluator,
    eval_type: GreenKernelEvalType,
    npoints: usize,
    sources: *const c_void,
    targets: *const c_void,
    result: *mut c_void,
) {
    fn impl_assemble<T: RlstScalar>(
        kernel_p: *mut GreenKernelEvaluator,
        eval_type: GreenKernelEvalType,
        npoints: usize,
        sources: *const c_void,
        targets: *const c_void,
        result: *const c_void,
    ) {
        let kernel = green_kernel_inner::<T>(kernel_p);
        let range_count = kernel.range_component_count(eval_type);
        let dim = unsafe { green_kernel_space_dimension(kernel_p) as usize };
        let sources: &[T::Real] =
            unsafe { std::slice::from_raw_parts(sources as *const T::Real, npoints * dim) };
        let targets: &[T::Real] =
            unsafe { std::slice::from_raw_parts(targets as *const T::Real, npoints * dim) };
        let result: &mut [T] =
            unsafe { std::slice::from_raw_parts_mut(result as *mut T, npoints * range_count) };
        kernel.assemble_pairwise_st(eval_type, sources, targets, result);
    }

    assert!(!kernel_p.is_null());

    match green_kernel_get_ctype(kernel_p) {
        GreenKernelCType::F32 => {
            impl_assemble::<f32>(kernel_p, eval_type, npoints, sources, targets, result);
        }
        GreenKernelCType::F64 => {
            impl_assemble::<f64>(kernel_p, eval_type, npoints, sources, targets, result);
        }
        GreenKernelCType::C32 => {
            impl_assemble::<c32>(kernel_p, eval_type, npoints, sources, targets, result);
        }
        GreenKernelCType::C64 => {
            impl_assemble::<c64>(kernel_p, eval_type, npoints, sources, targets, result);
        }
    }
}

/// Return the range component count.
///
/// # Safety
/// Pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn green_kernel_range_component_count(
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

/// Return the domain component count.
///
/// # Safety
/// Pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn green_kernel_domain_component_count(
    kernel_p: *mut GreenKernelEvaluator,
) -> u32 {
    assert!(!kernel_p.is_null());
    match green_kernel_get_ctype(kernel_p) {
        GreenKernelCType::F32 => {
            green_kernel_inner::<f32>(kernel_p).domain_component_count() as u32
        }
        GreenKernelCType::F64 => {
            green_kernel_inner::<f64>(kernel_p).domain_component_count() as u32
        }
        GreenKernelCType::C32 => {
            green_kernel_inner::<c32>(kernel_p).domain_component_count() as u32
        }
        GreenKernelCType::C64 => {
            green_kernel_inner::<c64>(kernel_p).domain_component_count() as u32
        }
    }
}

/// Return the space dimension.
///
/// # Safety
/// Pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn green_kernel_space_dimension(kernel_p: *mut GreenKernelEvaluator) -> u32 {
    assert!(!kernel_p.is_null());
    match green_kernel_get_ctype(kernel_p) {
        GreenKernelCType::F32 => green_kernel_inner::<f32>(kernel_p).space_dimension() as u32,
        GreenKernelCType::F64 => green_kernel_inner::<f64>(kernel_p).space_dimension() as u32,
        GreenKernelCType::C32 => green_kernel_inner::<c32>(kernel_p).space_dimension() as u32,
        GreenKernelCType::C64 => green_kernel_inner::<c64>(kernel_p).space_dimension() as u32,
    }
}

/// Evaluate the Greens function for a single source/target pair.
///
/// # Safety
/// Pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn greens_fct(
    kernel_p: *mut GreenKernelEvaluator,
    eval_type: GreenKernelEvalType,
    source: *const c_void,
    target: *const c_void,
    result: *mut c_void,
) {
    fn impl_greens_fct<T: RlstScalar>(
        kernel_p: *mut GreenKernelEvaluator,
        eval_type: GreenKernelEvalType,
        source: *const c_void,
        target: *const c_void,
        result: *mut c_void,
    ) {
        assert!(!kernel_p.is_null());
        let dim = unsafe { green_kernel_space_dimension(kernel_p) as usize };
        let range_components =
            unsafe { green_kernel_range_component_count(kernel_p, eval_type) as usize };
        let source = unsafe { std::slice::from_raw_parts(source as *const T::Real, dim) };
        let target = unsafe { std::slice::from_raw_parts(target as *const T::Real, dim) };
        let result = unsafe { std::slice::from_raw_parts_mut(result as *mut T, range_components) };
        green_kernel_inner::<T>(kernel_p).greens_fct(eval_type, source, target, result);
    }
    match green_kernel_get_ctype(kernel_p) {
        GreenKernelCType::F32 => {
            impl_greens_fct::<f32>(kernel_p, eval_type, source, target, result);
        }
        GreenKernelCType::F64 => {
            impl_greens_fct::<f64>(kernel_p, eval_type, source, target, result);
        }
        GreenKernelCType::C32 => {
            impl_greens_fct::<c32>(kernel_p, eval_type, source, target, result);
        }
        GreenKernelCType::C64 => {
            impl_greens_fct::<c64>(kernel_p, eval_type, source, target, result);
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_create_laplace_evaluator() {
        let evaluator = green_kernel_laplace_3d_alloc(GreenKernelCType::F32);
        assert_eq!(GreenKernelCType::F32, unsafe {
            green_kernel_get_ctype(evaluator)
        });

        unsafe { green_kernel_free(evaluator) };
    }
}
