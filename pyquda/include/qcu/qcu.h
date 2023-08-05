#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QcuParam_s {
  int lattice_size[4];
} QcuParam;

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param);
void dslashQcuEO(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int even_odd);
#ifdef __cplusplus
}
#endif
