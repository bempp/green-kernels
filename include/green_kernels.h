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

enum GreenKernelCType green_kernel_get_ctype(struct GreenKernelEvaluator *kernel_p);

void green_kernel_free(struct GreenKernelEvaluator *kernel_p);

struct GreenKernelEvaluator *green_kernel_laplace_3d_alloc(enum GreenKernelCType ctype);

struct GreenKernelEvaluator *green_kernel_modified_helmholtz_3d_alloc(enum GreenKernelCType ctype,
                                                                      double omega);

struct GreenKernelEvaluator *green_kernel_helmholtz_3d_alloc(enum GreenKernelCType ctype,
                                                             double wavenumber);

void green_kernel_evaluate(struct GreenKernelEvaluator *kernel_p,
                           enum GreenKernelEvalType eval_type,
                           uintptr_t nsources,
                           uintptr_t ntargets,
                           const void *sources,
                           const void *targets,
                           const void *charges,
                           void *result,
                           bool multithreaded);

void green_kernel_assemble(struct GreenKernelEvaluator *kernel_p,
                           enum GreenKernelEvalType eval_type,
                           uintptr_t nsources,
                           uintptr_t ntargets,
                           const void *sources,
                           const void *targets,
                           void *result,
                           bool multithreaded);

uint32_t green_kernel_range_component_count(struct GreenKernelEvaluator *kernel_p,
                                            enum GreenKernelEvalType eval_type);

uint32_t green_kernel_domain_component_count(struct GreenKernelEvaluator *kernel_p);

uint32_t green_kernel_space_dimension(struct GreenKernelEvaluator *kernel_p);

void greens_fct(struct GreenKernelEvaluator *kernel_p,
                enum GreenKernelEvalType eval_type,
                const void *source,
                const void *target,
                void *result);
