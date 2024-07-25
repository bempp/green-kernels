
#include<stdio.h>
#include<stdlib.h>
#include "green_kernels.h"

const int NSOURCES = 1000;
const int NTARGETS = 1000;

double drand() {
  return (double)rand() / RAND_MAX;
}

int main() {

  // Seed random numbers
  srand(0);

  bool multithreaded = true;

  // Instantiate a Laplace evaluator
  GreenKernelEvaluator* evaluator = green_kernel_laplace_3d_alloc(GreenKernelCType_F64);

  // Create random sources, targets, and charges
  double* sources = (double*) malloc(3 * NSOURCES * sizeof(double));
  double* targets = (double*) malloc(3 * NSOURCES * sizeof(double));
  double* charges = (double*) malloc(3 * NSOURCES * sizeof(double));
  double* result  = (double*) malloc(NSOURCES * NTARGETS * sizeof(double));

  for (int i = 0; i< 3 * NSOURCES; ++i) {
    sources[i] = drand();
  }
  
  for (int i = 0; i< 3 * NTARGETS; ++i) {
    targets[i] = drand();
  }

  for (int i = 0; i< NSOURCES; ++i) {
    charges[i] = drand();
  }

  for (int i = 0; i < NTARGETS; ++i) {
    result[i] = 0;
  }

  green_kernel_assemble(evaluator, GreenKernelEvalType_Value, NSOURCES, NTARGETS, sources, targets, result, multithreaded);

  printf("The potential from the first source at the first target is: %e \n", result[0]);

  free(sources);
  free(targets);
  free(charges);
  free(result);
  
}

