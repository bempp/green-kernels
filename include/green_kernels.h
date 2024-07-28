#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum GreenKernelCType {
  GreenKernelCType_F32,
  GreenKernelCType_F64,
  GreenKernelCType_C32,
  GreenKernelCType_C64,
} GreenKernelCType;

/**
 * Evaluation Mode
 */
typedef enum GreenKernelEvalType {
  /**
   * Only values required
   */
  GreenKernelEvalType_Value,
  /**
   * Both values and derivatives required
   */
  GreenKernelEvalType_ValueDeriv,
} GreenKernelEvalType;

typedef struct GreenKernelEvaluator GreenKernelEvaluator;

/**
 * Return the type of the kernel.
 *
 * # Safety
 * Pointer must be valid.
 */
enum GreenKernelCType green_kernel_get_ctype(struct GreenKernelEvaluator *kernel_p);

/**
 * Free the kernel.
 *
 * # Safety
 * Pointer must be valid.
 */
void green_kernel_free(struct GreenKernelEvaluator *kernel_p);

/**
 * Create a new Laplace kernel.
 */
struct GreenKernelEvaluator *green_kernel_laplace_3d_alloc(enum GreenKernelCType ctype);

/**
 * Create a new Modified Helmholtz kernel.
 */
struct GreenKernelEvaluator *green_kernel_modified_helmholtz_3d_alloc(enum GreenKernelCType ctype,
                                                                      double omega);

/**
 * Create a new Helmholtz kernel.
 */
struct GreenKernelEvaluator *green_kernel_helmholtz_3d_alloc(enum GreenKernelCType ctype,
                                                             double wavenumber);

/**
 * Evaluate a kernel.
 *
 * # Safety
 * Pointer must be valid.
 */
void green_kernel_evaluate(struct GreenKernelEvaluator *kernel_p,
                           enum GreenKernelEvalType eval_type,
                           uintptr_t nsources,
                           uintptr_t ntargets,
                           const void *sources,
                           const void *targets,
                           const void *charges,
                           void *result,
                           bool multithreaded);

/**
 * Assemble a kernel.
 *
 * # Safety
 * Pointer must be valid.
 */
void green_kernel_assemble(struct GreenKernelEvaluator *kernel_p,
                           enum GreenKernelEvalType eval_type,
                           uintptr_t nsources,
                           uintptr_t ntargets,
                           const void *sources,
                           const void *targets,
                           void *result,
                           bool multithreaded);

/**
 * Pairwise assembly of sources and targets.
 *
 * # Safety
 * Pointer must be valid.
 */
void green_kernel_pairwise_assemble(struct GreenKernelEvaluator *kernel_p,
                                    enum GreenKernelEvalType eval_type,
                                    uintptr_t npoints,
                                    const void *sources,
                                    const void *targets,
                                    void *result);

/**
 * Return the range component count.
 *
 * # Safety
 * Pointer must be valid.
 */
uint32_t green_kernel_range_component_count(struct GreenKernelEvaluator *kernel_p,
                                            enum GreenKernelEvalType eval_type);

/**
 * Return the domain component count.
 *
 * # Safety
 * Pointer must be valid.
 */
uint32_t green_kernel_domain_component_count(struct GreenKernelEvaluator *kernel_p);

/**
 * Return the space dimension.
 *
 * # Safety
 * Pointer must be valid.
 */
uint32_t green_kernel_space_dimension(struct GreenKernelEvaluator *kernel_p);

/**
 * Evaluate the Greens function for a single source/target pair.
 *
 * # Safety
 * Pointer must be valid.
 */
void greens_fct(struct GreenKernelEvaluator *kernel_p,
                enum GreenKernelEvalType eval_type,
                const void *source,
                const void *target,
                void *result);
