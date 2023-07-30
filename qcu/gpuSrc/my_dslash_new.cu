#include "qcu.h"
#include <cstdio>
#include <time.h>

#define NC 3
#define ND 4
#define NS 4
#define BLOCK_SIZE 256

// static void* gpu_gauge;
// static void* gpu_fermion_out;
// static void* gpu_fermion_in;


#define checkCudaErrors(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                err, cudaGetErrorString(err), __FILE__, __LINE__); \
                exit(-1); \
        }\
    }
// #define checkCudaErrors(err) { \
//     if (err != cudaSuccess) { \
//         fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
//                 err, cudaGetErrorString(err), __FILE__, __LINE__); \
//                 return(-1); \
//         }\
//     }
#define getVecAddr(origin, x, y, z, t, Lx, Ly, Lz, Lt)  \
    ((origin) + (t*(Lz*Ly*Lx) + z*(Ly*Lx) + y*Lx + x) * NS * NC)
#define getGaugeAddr(origin, direction, x, y, z, t, Lx, Ly, Lz, Lt) \
    ((origin) + (direction * (Lt*Lz*Ly*Lx) + t*(Lz*Ly*Lx) + z*(Ly*Lx) + y*Lx + x) * NC * NC)

class Complex {
private:
    double real_;
    double imag_;
public:
    __device__ __host__
    Complex(double real, double imag) : real_(real), imag_(imag) { }
    __device__ __host__
    Complex() : real_(0), imag_(0) {}
    __device__ __host__
    Complex(const Complex& complex) : real_(complex.real_), imag_(complex.imag_){}
    __device__ __host__
    void setImag(double imag) { imag_ = imag; }
    __device__ __host__
    void setReal(double real) { real_ = real; }
    __device__ __host__
    double real() const { return real_; }
    __device__ __host__
    double imag() const { return imag_; }

    __device__ __host__
    Complex& operator= (const Complex& complex) {
        real_ = complex.real_;
        imag_ = complex.imag_;
        return *this;
    }
    __device__ __host__
    Complex& operator= (double rhs) {
        real_ = rhs;
        imag_ = 0;
        return *this;
    }
    __device__ __host__
    Complex operator+(const Complex& complex) const {
        return Complex(real_+complex.real_, imag_+complex.imag_);
    }
    __device__ __host__
    Complex operator-(const Complex& complex) const {
        return Complex(real_-complex.real_, imag_-complex.imag_);
    }
    __device__ __host__
    Complex operator-() const{
        return Complex(-real_, -imag_);
    }
    __device__ __host__
    Complex operator*(const Complex& rhs) const {
        return Complex(real_ * rhs.real_ - imag_ * rhs.imag_, real_ * rhs.imag_ + imag_ * rhs.real_);
    }
    __device__ __host__
    Complex& operator*=(const Complex& rhs) {
        real_ = real_ * rhs.real_ - imag_ * rhs.imag_;
        imag_ = real_ * rhs.imag_ + imag_ * rhs.real_;
        return *this;
    }

    __device__ __host__
    Complex& operator+=(const Complex& rhs) {
        real_ += rhs.real_;
        imag_ += rhs.imag_;
        return *this;
    }

    __device__ __host__
    Complex& operator-=(const Complex& rhs) {
        real_ -= rhs.real_;
        imag_ -= rhs.imag_;
        return *this;
    }

    __device__ __host__
    Complex& clear2Zero() {
        real_ = 0;
        imag_ = 0;
        return *this;
    }
    __device__ __host__
    Complex conj() {
        return Complex(real_, -imag_);
    }
    __device__ __host__
    bool operator==(const Complex& rhs) {
        return real_ == rhs.real_ && imag_ == rhs.imag_;
    }
    __device__ __host__
    bool operator!=(const Complex& rhs) {
        return real_ != rhs.real_ || imag_ != rhs.imag_;
    }
};

__global__
void gpuDslash(void* U_ptr, void* a_ptr, void* b_ptr, int Lx, int Ly, int Lz, int Lt) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int t = thread / (Lx * Ly * Lz);
    int z = (thread % (Lx * Ly * Lz)) / (Lx * Ly);
    int y = (thread % (Lx * Ly)) / Lx;
    int x = thread % Lx;

    int pos_x, pos_y, pos_z, pos_t;
    Complex *u;
    Complex *res;
    Complex *dest;
    Complex u_temp[NC * NC];            // for GPU
    Complex res_temp[NS * NC];          // for GPU
    Complex dest_temp[NS * NC];         // for GPU
    Complex temp;

    for (int i = 0; i < NS*NC; i++) {
        dest_temp[i].clear2Zero();
    }
    // \mu = 1
    pos_x = (x+1)%Lx;
    pos_y = y;
    pos_z = z;
    pos_t = t;
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 0, x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC * NC; i++) {
        u_temp[i] = u[i];
    }
    res = getVecAddr(static_cast<Complex*>(a_ptr), pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
    dest = getVecAddr(static_cast<Complex*>(b_ptr), x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC; i++) {
        for (int j = 0; j < NC; j++) {
            // first row vector with col vector
            temp = (res_temp[0*NC+j] - res_temp[3*NC+j] * Complex(0, 1)) * u_temp[i*NC+j];
            dest_temp[0*3+i] += temp;
            dest_temp[3*3+i] += temp * Complex(0,1);
            // second row vector with col vector
            temp = (res_temp[1*NC+j] - res_temp[2*NC+j] * Complex(0, 1)) * u_temp[i*NC+j];
            dest_temp[1*3+i] += temp;
            dest_temp[2*3+i] += temp * Complex(0,1);
        }
    }
    pos_x = (x+Lx-1)%Lx;
    pos_y = y;
    pos_z = z;
    pos_t = t;
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 0, pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC * NC; i++) {
        u_temp[i] = u[i];
    }
    res = getVecAddr(static_cast<Complex*>(a_ptr), pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
    dest = getVecAddr(static_cast<Complex*>(b_ptr), x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC; i++) {
        for (int j = 0; j < NC; j++) {
            // first row vector with col vector
            temp = (res_temp[0*NC+j] + res_temp[3*NC+j] * Complex(0,1)) * u_temp[j*NC+i].conj();   // transpose and conj
            dest_temp[0*3+i] += temp;
            dest_temp[3*3+i] += temp * Complex(0, -1);
            // second row vector with col vector
            temp = (res_temp[1*NC+j] + res_temp[2*NC+j] * Complex(0,1)) * u_temp[j*NC+i].conj();   // transpose and conj
            dest_temp[1*3+i] += temp;
            dest_temp[2*3+i] += temp * Complex(0, -1);
        }
    }
    // \mu = 2
    // linear combine
    pos_x = x;
    pos_y = (y+1)%Ly;
    pos_z = z;
    pos_t = t;
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 1, x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC * NC; i++) {
        u_temp[i] = u[i];
    }
    res = getVecAddr(static_cast<Complex*>(a_ptr), pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
    dest = getVecAddr(static_cast<Complex*>(b_ptr), x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC; i++) {
        for (int j = 0; j < NC; j++) {
            // first row vector with col vector
            temp = (res_temp[0*NC+j] + res_temp[3*NC+j]) * u_temp[i*NC+j];
            dest_temp[0*3+i] += temp;
            dest_temp[3*3+i] += temp;
            // second row vector with col vector
            temp = (res_temp[1*NC+j] - res_temp[2*NC+j]) * u_temp[i*NC+j];
            dest_temp[1*3+i] += temp;
            dest_temp[2*3+i] += -temp;
        }
    }
    pos_x = x;
    pos_y = (y+Ly-1)%Ly;
    pos_z = z;
    pos_t = t;
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 1, pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC * NC; i++) {
        u_temp[i] = u[i];
    }
    res = getVecAddr(static_cast<Complex*>(a_ptr), pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
    dest = getVecAddr(static_cast<Complex*>(b_ptr), x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC; i++) {
        for (int j = 0; j < NC; j++) {
            // first row vector with col vector
            temp = (res_temp[0*NC+j] - res_temp[3*NC+j]) * u_temp[j*NC+i].conj();   // transpose and conj
            dest_temp[0*3+i] += temp;
            dest_temp[3*3+i] += -temp;
            // second row vector with col vector
            temp = (res_temp[1*NC+j] + res_temp[2*NC+j]) * u_temp[j*NC+i].conj();   // transpose and conj
            dest_temp[1*3+i] += temp;
            dest_temp[2*3+i] += temp;
        }
    }
    // \mu = 3
    pos_x = x;
    pos_y = y;
    pos_z = (z+1)%Lz;
    pos_t = t;
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 2, x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC * NC; i++) {
        u_temp[i] = u[i];
    }
    res = getVecAddr(static_cast<Complex*>(a_ptr), pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
    dest = getVecAddr(static_cast<Complex*>(b_ptr), x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC; i++) {
        for (int j = 0; j < NC; j++) {
            // first row vector with col vector
            temp = (res_temp[0*NC+j] - res_temp[2*NC+j] * Complex(0, 1)) * u_temp[i*NC+j];
            dest_temp[0*3+i] += temp;
            dest_temp[2*3+i] += temp * Complex(0, 1);
            // second row vector with col vector
            temp = (res_temp[1*NC+j] + res_temp[3*NC+j] * Complex(0,1)) * u_temp[i*NC+j];
            dest_temp[1*3+i] += temp;
            dest_temp[3*3+i] += temp * Complex(0, -1);
        }
    }
    pos_x = x;
    pos_y = y;
    pos_z = (z+Lz-1)%Lz;
    pos_t = t;
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 2, pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC * NC; i++) {
        u_temp[i] = u[i];
    }
    res = getVecAddr(static_cast<Complex*>(a_ptr), pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
    dest = getVecAddr(static_cast<Complex*>(b_ptr), x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC; i++) {
        for (int j = 0; j < NC; j++) {
            // first row vector with col vector
            temp = (res_temp[0*NC+j] + res_temp[2*NC+j] * Complex(0, 1)) * u_temp[j*NC+i].conj();   // transpose and conj
            dest_temp[0*3+i] += temp;
            dest_temp[2*3+i] += temp * Complex(0, -1);
            // second row vector with col vector
            temp = (res_temp[1*NC+j] - res_temp[3*NC+j] * Complex(0, 1)) * u_temp[j*NC+i].conj();   // transpose and conj
            dest_temp[1*3+i] += temp;
            dest_temp[3*3+i] += temp * Complex(0, 1);
        }
    }
    // \mu = 4
    pos_x = x;
    pos_y = y;
    pos_z = z;
    pos_t = (t+1)%Lt;
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 3, x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC * NC; i++) {
        u_temp[i] = u[i];
    }
    res = getVecAddr(static_cast<Complex*>(a_ptr), pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
    dest = getVecAddr(static_cast<Complex*>(b_ptr), x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC; i++) {
        for (int j = 0; j < NC; j++) {
            // first row vector with col vector
            temp = (res_temp[0*NC+j] - res_temp[2*NC+j]) * u_temp[i*NC+j];
            dest_temp[0*3+i] += temp;
            dest_temp[2*3+i] += -temp;
            // second row vector with col vector
            temp = (res_temp[1*NC+j] - res_temp[3*NC+j]) * u_temp[i*NC+j];
            dest_temp[1*3+i] += temp;
            dest_temp[3*3+i] += -temp;
        }
    }
    pos_x = x;
    pos_y = y;
    pos_z = z;
    pos_t = (t+Lt-1)%Lt;
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 3, pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC * NC; i++) {
        u_temp[i] = u[i];
    }
    res = getVecAddr(static_cast<Complex*>(a_ptr), pos_x, pos_y, pos_z, pos_t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
    dest = getVecAddr(static_cast<Complex*>(b_ptr), x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NC; i++) {
        for (int j = 0; j < NC; j++) {
            // first row vector with col vector
            temp = (res_temp[0*NC+j] + res_temp[2*NC+j]) * u_temp[j*NC+i].conj();   // transpose and conj
            dest_temp[0*3+i] += temp;
            dest_temp[2*3+i] += temp;
            // second row vector with col vector
            temp = (res_temp[1*NC+j] + res_temp[3*NC+j]) * u_temp[j*NC+i].conj();   // transpose and conj
            dest_temp[1*3+i] += temp;
            dest_temp[3*3+i] += temp;
        }
    }
    // end, copy result to dest
    for (int i = 0; i < NS * NC; i++) {
        dest[i] = dest_temp[i];
    }
}

// int prepare(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, size_t *gauge_ptr, size_t *fermion_in_ptr, size_t *fermion_out_ptr) {
//     int Lx = param->lattice_size[0];
//     int Ly = param->lattice_size[1];
//     int Lz = param->lattice_size[2];
//     int Lt = param->lattice_size[3];
//     unsigned long u_size = ND * Lt * Lz * Ly * Lx * NC * NC * sizeof(Complex);
//     unsigned long vec_size = Lt * Lz * Ly * Lx * NC * NC * sizeof(Complex);

//     checkCudaErrors(cudaMalloc(&gpu_gauge, u_size));
//     checkCudaErrors(cudaMalloc(&gpu_fermion_out, vec_size));
//     checkCudaErrors(cudaMalloc(&gpu_fermion_in, vec_size));



//     return 0; // success
// }

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param) {
    clock_t start, end;
    clock_t big_start, big_end;
    int Lx = param->lattice_size[0];
    int Ly = param->lattice_size[1];
    int Lz = param->lattice_size[2];
    int Lt = param->lattice_size[3];

    void* d_u;
    void* d_a;
    void* d_b;
    unsigned long u_size = ND * Lt * Lz * Ly * Lx * NC * NC * sizeof(Complex);
    unsigned long vec_size = Lt * Lz * Ly * Lx * NS * NC * sizeof(Complex);
    int space = Lx * Ly * Lz * Lt;

    big_start = clock();
    checkCudaErrors(cudaMalloc(&d_u, u_size));
    checkCudaErrors(cudaMalloc(&d_a, vec_size));
    checkCudaErrors(cudaMalloc(&d_b, vec_size));

    checkCudaErrors(cudaMemcpy(d_u, gauge, u_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_a, fermion_in, vec_size, cudaMemcpyHostToDevice));

    dim3 gridDim(space/BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    start = clock();
    // 改为GPU版本 -- kernel function
    gpuDslash<<<gridDim, blockDim>>>(d_u, d_a, d_b, Lx, Ly, Lz, Lt);
    cudaError_t err = cudaGetLastError();
    checkCudaErrors(err);
    // 数据同步
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    // 执行结果不改变U和a，所以这里不去进行u和a的cpy
    checkCudaErrors(cudaMemcpy(fermion_out, d_b, vec_size, cudaMemcpyDeviceToHost));
    // free memory
    checkCudaErrors(cudaFree(d_u));
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    big_end = clock();
    printf("total time: (with malloc free memcpy) : %lf\n", (double)(big_end - big_start) / CLOCKS_PER_SEC);
    printf("total time: (without malloc free memcpy) : %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
}
