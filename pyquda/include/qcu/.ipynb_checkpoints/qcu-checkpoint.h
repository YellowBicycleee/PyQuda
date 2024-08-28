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
void initGridSize(QcuGrid_t* grid, QcuParam* p_param, void* gauge, void* fermion_in, void* fermion_out); // initialize the struct of grid
void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);

#ifdef __cplusplus
}
#endif
