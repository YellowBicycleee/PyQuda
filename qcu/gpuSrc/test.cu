#include <stdio.h>
#include <stdlib.h>
#include "qcu.h"
int main () {

    FILE* fp = fopen("./input.txt", "r");

    const size_t Lt = 32;
    const size_t Lz = 16;
    const size_t Ly = 16;
    const size_t Lx = 16;
    const size_t Ns = 4;
    const size_t Nc = 3;
    const size_t Nd = 4;

    double* U = (double*)malloc((Nd * Lt * Lz * Ly * Lx * Nc * Nc * 2)*sizeof(double));
    double* a = (double*)malloc((Lt * Lz * Ly * Lx * Ns * Nc * 2)*sizeof(double));
    double* b = (double*)malloc((Lt * Lz * Ly * Lx * Ns * Nc * 2)*sizeof(double));
    for (int i = 0; i < 9 * 2; i++) {
        fscanf(fp, "%lf", &(U[i]));
    }
    for (int i = 0; i < 9 * 2; i++) {
        printf("%lf ", U[i]);
    }
    for (int i = 0; i < 6; i++)
        a[2 * i] = 1;

    QcuParam param;
    param.lattice_size[0] = Lx;
    param.lattice_size[1] = Ly;
    param.lattice_size[2] = Lz;
    param.lattice_size[3] = Lt;
    dslashQcuEO((void*)(b), (void*)(a), (void*)(U), &param, 0);
    double* start1 = (double*)(b) + 24;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 6; j++) {
            // cout << start1[i * 6 + j] << " ";
            printf("%lf\t", start1[i*6+j]);
        }
        printf("\n");
    }
    printf("\n");

    free(U);
    free(a);
    free(b);

    fclose(fp);
    return 0;
}