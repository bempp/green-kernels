#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum GreenKernelCType {
  GreenKernelF32,
  GreenKernelF64,
  GreenKernelC32,
  GreenKernelC64,
} GreenKernelCType;

/**
 * Evaluation Mode
 */
typedef enum GreenKernelEvalType {
  /**
   * Only values required
   */
  Value,
  /**
   * Both values and derivatives required
   */
  ValueDeriv,
} GreenKernelEvalType;

typedef struct GreenKernelEvaluator GreenKernelEvaluator;

enum GreenKernelCType green_kernel_get_ctype(struct GreenKernelEvaluator *kernel_p);

void green_kernel_free(struct GreenKernelEvaluator *kernel_p);

struct GreenKernelEvaluator *green_kernel_laplace_3d_alloc(enum GreenKernelCType ctype);

void green_kernel_evaluate(struct GreenKernelEvaluator *kernel_p,
                           enum GreenKernelEvalType eval_type,
                           uintptr_t nsources,
                           uintptr_t ntargets,
                           const void *sources,
                           const void *targets,
                           const void *charges,
                           void *result,
                           uintptr_t dim,
                           bool multithreaded);

uint32_t green_kernel_range_component_count(struct GreenKernelEvaluator *kernel_p,
                                            enum GreenKernelEvalType eval_type);
