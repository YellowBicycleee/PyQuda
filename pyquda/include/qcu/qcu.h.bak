#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QcuParam_s {
  int lattice_size[4];
} QcuParam;

typedef struct QcuGrid_t {
  int grid_size[4];
} QcuGrid_t;
void initGridSize(QcuGrid_t* grid, QcuParam* p_param, void* gauge, void* fermion_in, void* fermion_out);
void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);

void fullDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int dagger_flag);
void cg_inverter(void *x_vector, void *b_vector, void *gauge, QcuParam *param, double p_max_prec, double p_kappa);

void loadQcuGauge(void* gauge, QcuParam *param);

#ifdef __cplusplus
}
#endif

