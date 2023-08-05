#include "qcu.h"
#include <cstdio>
#include <time.h>
#include <cmath>
#include <assert.h>
#define NC 3
#define ND 4
#define NS 4
#define BLOCK_SIZE 128

#define checkCudaErrors(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                err, cudaGetErrorString(err), __FILE__, __LINE__); \
                exit(-1); \
        }\
    }

#define getVecAddr(origin, x, y, z, t, Lx, Ly, Lz, Lt)  \
    ((origin) + ((((t) * (Lz) + (z)) * (Ly) + (y))*(Lx) + (x)) * NS * NC)   // 9 times
#define getGaugeAddr(origin, direction, x, y, z, t, Lx, Ly, Lz, Lt, even_odd) \
    ((origin) + (direction) * (Lt) * (Lz) * (Ly) * (Lx) * 2 * NC * NC + (even_odd) * ((Lt) * (Lz) * (Ly) * (Lx)) + ((((t) * (Lz) + (z)) * (Ly) + (y))*(Lx) + (x)) * NC * NC)


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
    double norm2() {
        return sqrt(real_ * real_ + imag_ * imag_);
    }
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
    Complex operator/ (const double& rhs) {
        return Complex(real_/rhs, imag_/rhs);
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

// even_odd == 0---->even else odd
// transfer 
__global__
void gpuDslash(void* U_ptr, void* a_ptr, void* b_ptr, int Lx, int Ly, int Lz, int Lt, int even_odd) {
    // Lx >> 2;
    assert(even_odd == 0 || even_odd == 1);
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    // int odd_Lx = Lx;

    Lx >>= 1;
    int t = thread / (Lx * Ly * Lz);
    thread -= t * (Lx * Ly * Lz);
    int z = thread / (Lx * Ly);
    thread -= z * (Lx * Ly);
    int y = thread / Lx;
    int x = thread - y * Lx;
    // int old_Lx = Lx;
    // int sub_vol = Lt * Lz * Ly * Lx >> 2;


    int eo = (t+z+y) % 2;
    int pos_x;
    Complex *u;
    Complex *res;
    Complex *dest;
    Complex u_temp[NC * NC];            // for GPU
    Complex res_temp[NS * NC];          // for GPU
    Complex dest_temp[NS * NC];         // for GPU
    Complex u_last_line[NC];
    // double norm;

    Complex temp;
    for (int i = 0; i < NS*NC; i++) {
        dest_temp[i].clear2Zero();
    }
    dest = getVecAddr(static_cast<Complex*>(b_ptr), x, y, z, t, Lx, Ly, Lz, Lt);


    // \mu = 1
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 0, x, y, z, t, Lx, Ly, Lz, Lt, even_odd);
    // #pragma unroll
    for (int i = 0; i < 2 * NC; i++) {
        u_temp[i] = u[i];
    }
    u_last_line[0] = u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4];
    u_last_line[1] = u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5];
    u_last_line[2] = u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3];
    // norm = sqrt(u_last_line[0].norm2() * u_last_line[0].norm2() + u_last_line[1].norm2() * u_last_line[1].norm2() + u_last_line[2].norm2() * u_last_line[2].norm2());
    u_temp[6] = u_last_line[0].conj();// / norm;
    u_temp[7] = u_last_line[1].conj();// / norm;
    u_temp[8] = u_last_line[2].conj();// / norm;
    pos_x = (1 & ~(even_odd ^ eo)) * x +  (even_odd ^ eo) * ((x+1)%Lx);
    res = getVecAddr(static_cast<Complex*>(a_ptr), pos_x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
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
    pos_x = (1 & ~(even_odd ^ eo)) * ((x-1+Lx) % Lx) + (even_odd ^ eo) * x;
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 0, pos_x, y, z, t, Lx, Ly, Lz, Lt, 1-even_odd);
    // for (int i = 0; i < NC * NC; i++) {
    //     u_temp[i] = u[i];
    // }
    for (int i = 0; i < 2 * NC; i++) {
        u_temp[i] = u[i];
    }
    u_last_line[0] = u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4];
    u_last_line[1] = u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5];
    u_last_line[2] = u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3];
    // norm = sqrt(u_last_line[0].norm2() * u_last_line[0].norm2() + u_last_line[1].norm2() * u_last_line[1].norm2() + u_last_line[2].norm2() * u_last_line[2].norm2());
    u_temp[6] = u_last_line[0].conj(); // / norm;
    u_temp[7] = u_last_line[1].conj(); // / norm;
    u_temp[8] = u_last_line[2].conj(); // / norm;

    res = getVecAddr(static_cast<Complex*>(a_ptr), pos_x, y, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
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
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 1, x, y, z, t, Lx, Ly, Lz, Lt, even_odd);
    // for (int i = 0; i < NC * NC; i++) {
    //     u_temp[i] = u[i];
    // }
    for (int i = 0; i < 2 * NC; i++) {
        u_temp[i] = u[i];
    }
    u_last_line[0] = u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4];
    u_last_line[1] = u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5];
    u_last_line[2] = u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3];
    // norm = sqrt(u_last_line[0].norm2() * u_last_line[0].norm2() + u_last_line[1].norm2() * u_last_line[1].norm2() + u_last_line[2].norm2() * u_last_line[2].norm2());
    u_temp[6] = u_last_line[0].conj(); // / norm;
    u_temp[7] = u_last_line[1].conj(); // / norm;
    u_temp[8] = u_last_line[2].conj(); // / norm;
    res = getVecAddr(static_cast<Complex*>(a_ptr), x, (y+1)%Ly, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
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
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 1, x, (y+Ly-1)%Ly, z, t, Lx, Ly, Lz, Lt, 1-even_odd);
    // for (int i = 0; i < NC * NC; i++) {
    //     u_temp[i] = u[i];
    // }
    for (int i = 0; i < 2 * NC; i++) {
        u_temp[i] = u[i];
    }
    u_last_line[0] = u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4];
    u_last_line[1] = u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5];
    u_last_line[2] = u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3];
    // norm = sqrt(u_last_line[0].norm2() * u_last_line[0].norm2() + u_last_line[1].norm2() * u_last_line[1].norm2() + u_last_line[2].norm2() * u_last_line[2].norm2());
    u_temp[6] = u_last_line[0].conj(); // / norm;
    u_temp[7] = u_last_line[1].conj(); // / norm;
    u_temp[8] = u_last_line[2].conj(); // / norm;

    res = getVecAddr(static_cast<Complex*>(a_ptr), x, (y+Ly-1)%Ly, z, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
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
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 2, x, y, z, t, Lx, Ly, Lz, Lt, even_odd);
    // for (int i = 0; i < NC * NC; i++) {
    //     u_temp[i] = u[i];
    // }
    for (int i = 0; i < 2 * NC; i++) {
        u_temp[i] = u[i];
    }
    u_last_line[0] = u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4];
    u_last_line[1] = u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5];
    u_last_line[2] = u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3];
    // norm = sqrt(u_last_line[0].norm2() * u_last_line[0].norm2() + u_last_line[1].norm2() * u_last_line[1].norm2() + u_last_line[2].norm2() * u_last_line[2].norm2());
    u_temp[6] = u_last_line[0].conj(); // norm;
    u_temp[7] = u_last_line[1].conj(); // norm;
    u_temp[8] = u_last_line[2].conj(); // norm;
    res = getVecAddr(static_cast<Complex*>(a_ptr), x, y, (z+1)%Lz, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
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
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 2, x, y, (z+Lz-1)%Lz, t, Lx, Ly, Lz, Lt, 1-even_odd);
    // for (int i = 0; i < NC * NC; i++) {
    //     u_temp[i] = u[i];
    // }
    for (int i = 0; i < 2 * NC; i++) {
        u_temp[i] = u[i];
    }
    u_last_line[0] = u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4];
    u_last_line[1] = u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5];
    u_last_line[2] = u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3];
    // norm = sqrt(u_last_line[0].norm2() * u_last_line[0].norm2() + u_last_line[1].norm2() * u_last_line[1].norm2() + u_last_line[2].norm2() * u_last_line[2].norm2());
    u_temp[6] = u_last_line[0].conj(); // norm;
    u_temp[7] = u_last_line[1].conj(); // norm;
    u_temp[8] = u_last_line[2].conj(); // norm;
    res = getVecAddr(static_cast<Complex*>(a_ptr), x, y, (z+Lz-1)%Lz, t, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
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
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 3, x, y, z, t, Lx, Ly, Lz, Lt, even_odd);
    // for (int i = 0; i < NC * NC; i++) {
    //     u_temp[i] = u[i];
    // }
    for (int i = 0; i < 2 * NC; i++) {
        u_temp[i] = u[i];
    }
    u_last_line[0] = u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4];
    u_last_line[1] = u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5];
    u_last_line[2] = u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3];
    // norm = sqrt(u_last_line[0].norm2() * u_last_line[0].norm2() + u_last_line[1].norm2() * u_last_line[1].norm2() + u_last_line[2].norm2() * u_last_line[2].norm2());
    u_temp[6] = u_last_line[0].conj(); // norm;
    u_temp[7] = u_last_line[1].conj(); // norm;
    u_temp[8] = u_last_line[2].conj(); // norm;
    res = getVecAddr(static_cast<Complex*>(a_ptr), x, y, z, (t+1)%Lt, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
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
    u = getGaugeAddr(static_cast<Complex*>(U_ptr), 3, x, y, z, (t+Lt-1)%Lt, Lx, Ly, Lz, Lt, 1-even_odd);
    // #pragma unroll
    // for (int i = 0; i < NC * NC; i++) {
    //     u_temp[i] = u[i];
    // }
    for (int i = 0; i < 2 * NC; i++) {
        u_temp[i] = u[i];
    }
    u_last_line[0] = u_temp[1] * u_temp[5] - u_temp[2] * u_temp[4];
    u_last_line[1] = u_temp[2] * u_temp[3] - u_temp[0] * u_temp[5];
    u_last_line[2] = u_temp[0] * u_temp[4] - u_temp[1] * u_temp[3];
    // norm = sqrt(u_last_line[0].norm2() * u_last_line[0].norm2() + u_last_line[1].norm2() * u_last_line[1].norm2() + u_last_line[2].norm2() * u_last_line[2].norm2());
    u_temp[6] = u_last_line[0].conj(); // / norm;
    u_temp[7] = u_last_line[1].conj(); // / norm;
    u_temp[8] = u_last_line[2].conj(); // / norm;
    res = getVecAddr(static_cast<Complex*>(a_ptr), x, y, z, (t+Lt-1)%Lt, Lx, Ly, Lz, Lt);
    for (int i = 0; i < NS * NC; i++) {
        res_temp[i] = res[i];
    }
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


void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param) {
    int even_odd = 0;   // waited to modify


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
    unsigned long vec_size = Lt * Lz * Ly * Lx * NS * NC * sizeof(Complex) >> 1;
    int space = Lx * Ly * Lz * Lt >> 1;

    big_start = clock();
    checkCudaErrors(cudaMalloc(&d_u, u_size));
    checkCudaErrors(cudaMalloc(&d_a, vec_size));
    checkCudaErrors(cudaMalloc(&d_b, vec_size));

    checkCudaErrors(cudaMemcpy(d_u, gauge, u_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_a, fermion_in, vec_size, cudaMemcpyHostToDevice));

    dim3 gridDim(space / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    start = clock();
    // kernel function
    gpuDslash<<<gridDim, blockDim>>>(d_u, d_a, d_b, Lx, Ly, Lz, Lt, even_odd);
    cudaError_t err = cudaGetLastError();
    checkCudaErrors(err);
    // sync
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    
    checkCudaErrors(cudaMemcpy(fermion_out, d_b, vec_size, cudaMemcpyDeviceToHost));
    // free memory
    checkCudaErrors(cudaFree(d_u));
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    big_end = clock();
    printf("total time: (with malloc free memcpy) : %lf\n", (double)(big_end - big_start) / CLOCKS_PER_SEC);
    printf("total time: (without malloc free memcpy) : %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
}
