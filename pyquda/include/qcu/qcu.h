#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QcuParam_s {
    int lattice_size[4];
} QcuParam;

typedef struct QcuGrid_t {
    int grid_size[4];
} QcuGrid;
void initGridSize(QcuGrid *grid, QcuParam *param, int n_color, int m_rhs, int inputFloatPrecision,
                  int dslashFloatPrecision);
void pushBackFermions(void *fermionOut, void *fermionIn);
void loadQcuGauge(void *gauge, int floatPrecision);  // double precision
void getDslash(int dslashType, double mass);  // dslash precision
void start_dslash(int parity, int daggerFlag);
void mat_Qcu(int daggerFlag);
void qcuInvert(int max_iteration, double p_max_prec);
void finalizeQcu();

// 奇偶预处理接口
void gauge_eo_precondition (void* prec_gauge, void* non_prec_gauge, int precision);
void gauge_reverse_eo_precondition(void* non_prec_gauge, void* prec_gauge, int precision);

// 文件读取接口
void read_gauge_from_file (void* gauge, const char* file_path_prefix);
#ifdef __cplusplus
}
#endif
