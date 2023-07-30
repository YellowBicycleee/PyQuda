#include "qcu.h"


#define NC 3
#define ND 4
#define NS 4

#define getVecAddr(origin, x, y, z, t, Lx, Ly, Lz, Lt)  \
    ((origin) + (t*(Lz*Ly*Lx) + z*(Ly*Lx) + y*Lx + x) * NS * NC)
#define getGaugeAddr(origin, direction, x, y, z, t, Lx, Ly, Lz, Lt) \
    ((origin) + (direction * (Lt*Lz*Ly*Lx) + t*(Lz*Ly*Lx) + z*(Ly*Lx) + y*Lx + x) * NC * NC)

class Complex {
private:
    double real_;
    double imag_;
public:
    Complex(double real, double imag) : real_(real), imag_(imag) { }
    Complex() : real_(0), imag_(0) {}
    Complex(const Complex& complex) : real_(complex.real_), imag_(complex.imag_){}
    void setImag(double imag) { imag_ = imag; }
    void setReal(double real) { real_ = real; }
    double real() const { return real_; }
    double imag() const { return imag_; }
    Complex& operator= (const Complex& complex) {
        real_ = complex.real_;
        imag_ = complex.imag_;
        return *this;
    }
    Complex& operator= (double rhs) {
        real_ = rhs;
        imag_ = 0;
        return *this;
    }
    Complex operator+(const Complex& complex) const {
        return Complex(real_+complex.real_, imag_+complex.imag_);
    }
    Complex operator-(const Complex& complex) const {
        return Complex(real_-complex.real_, imag_-complex.imag_);
    }
    Complex operator-() const{
        return Complex(-real_, -imag_);
    }
    Complex operator*(const Complex& rhs) const {
        return Complex(real_ * rhs.real_ - imag_ * rhs.imag_, real_ * rhs.imag_ + imag_ * rhs.real_);
    }
    Complex& operator*=(const Complex& rhs) {
        real_ = real_ * rhs.real_ - imag_ * rhs.imag_;
        imag_ = real_ * rhs.imag_ + imag_ * rhs.real_;
        return *this;
    }

    Complex& operator+=(const Complex& rhs) {
        real_ += rhs.real_;
        imag_ += rhs.imag_;
        return *this;
    }

    Complex& operator-=(const Complex& rhs) {
        real_ -= rhs.real_;
        imag_ -= rhs.imag_;
        return *this;
    }

    Complex& clear2Zero() {
        real_ = 0;
        imag_ = 0;
        return *this;
    }
    Complex conj() {
        return Complex(real_, -imag_);
    }
    bool operator==(const Complex& rhs) {
        return real_ == rhs.real_ && imag_ == rhs.imag_;
    }
    bool operator!=(const Complex& rhs) {
        return real_ != rhs.real_ || imag_ != rhs.imag_;
    }
};

// new serial code, expand \mu = 1,2,3,4
void newDslash(void* U_ptr, void* a_ptr, void* b_ptr, int Lx, int Ly, int Lz, int Lt) {
    int pos_x, pos_y, pos_z, pos_t;
    Complex *u;
    Complex *res;
    Complex *dest;
    Complex u_temp[NC * NC];   // for GPU
    Complex res_temp[NS * NC];          // for GPU
    Complex dest_temp[NS * NC];          // for GPU
    Complex temp;
    for (int t = 0; t < Lt; t++) {
        for (int z = 0; z < Lz; z++) {
            for (int y = 0; y < Ly; y++) {
                for (int x = 0; x < Lx; x++) {
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
            }
        }
    }
}

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param) {
    int Lx = param->lattice_size[0];
    int Ly = param->lattice_size[1];
    int Lz = param->lattice_size[2];
    int Lt = param->lattice_size[3];
    newDslash(gauge, fermion_in, fermion_out, Lx, Ly, Lz, Lt);
}
