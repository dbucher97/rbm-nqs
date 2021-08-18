/*
 * Copyright (c) 2021 David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */
#pragma once

#define LNCOSH_CUTOFF 16.7

#include <pfapack.h>

#include <Eigen/Dense>
#include <cmath>
#include <complex>

namespace math {

static const double ln2 = std::log(2);  ///< log(2) stored.

/**
 * @brief Calculates lncosh for a real value.
 *
 * @param x val
 *
 * @return ln(cosh(x))
 */
// static inline double lncosh(double x) {
//     x = std::abs(x);
//     // for x larger than the specified value, a linear approximation is more
//     // than sufficient
//     if (x > LNCOSH_CUTOFF) {
//         return x - ln2;
//     } else {
//         return std::log(std::cosh(x));
//     }
// }

/**
 * @brief Calculates lncosh for a complex value.
 *
 * @param x val
 *
 * @return ln(cosh(x))
 */
// static inline std::complex<double> lncosh(std::complex<double> x) {
//     const double xr = x.real();
//     const double xi = x.imag();

//     // Calculate the real lncosh
//     std::complex<double> ret = lncosh(xr);

//     // Calculate the complete lncosh with tanh, sin, which do not go to
//     // infinity with real values.
//     ret += std::log(
//         std::complex<double>(std::cos(xi), std::tanh(xr) * std::sin(xi)));

//     return ret;
// }

/**
 * @brief A `MatrixXcd` wrapper for lncosh, applying lncosh to all elements.
 *
 * @param x The reference to the matrix.
 *
 * @return MatrixXcd result.
 */

extern inline Eigen::ArrayXXcd lncosh(const Eigen::MatrixXcd& x) {
    Eigen::ArrayXXd xr = x.real().array();
    Eigen::ArrayXXd xi = x.imag().array();
    Eigen::ArrayXXcd ret;

    ret = (xr.abs() < LNCOSH_CUTOFF).select(xr.cosh().log(), xr - ln2);

    ret +=
        (xi.cos() + std::complex<double>(0, 1.) * xr.tanh() * xi.sin()).log();
    return ret;
}

extern inline Eigen::ArrayXXcd cosh2(const Eigen::MatrixXcd& x) {
    return (1 + x.real().array().pow(2) / 2) * x.imag().array().cos() +
           std::complex<double>(0, 1) * x.real().array() *
               x.imag().array().sin();
}

extern inline Eigen::ArrayXXcd tanh2(const Eigen::MatrixXcd& x) {
    return (x.real().array() * x.imag().array().cos() +
            std::complex<double>(0, 1) * (1 + x.real().array().pow(2) / 2) *
                x.imag().array().sin()) /
           cosh2(x).array();
}

extern inline Eigen::ArrayXXcd cosh1(const Eigen::MatrixXcd& x) {
    return x.array().cosh();
}

extern inline Eigen::ArrayXXcd tanh1(const Eigen::MatrixXcd& x) {
    Eigen::MatrixXcd r = x.array().tanh();
    return r.array().isFinite().select(r, 0.);
}

extern inline std::complex<double> pfaffian(Eigen::MatrixXcd& x) {
    std::complex<double> ret;
    skpfa_z(x.cols(), x.data(), &ret, "L", "P");
    return ret;
}

extern inline std::complex<double> pfaffian10(Eigen::MatrixXcd& x, int& exp) {
    std::complex<double> ret[2];
    skpf10_z(x.cols(), x.data(), ret, "L", "P");
    exp = (int)std::real(ret[1]);
    return ret[0];
}

}  // namespace math
