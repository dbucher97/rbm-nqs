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
static inline double lncosh(double x) {
    x = std::abs(x);
    // for x larger than the specified value, a linear approximation is more
    // than sufficient
    if (x > LNCOSH_CUTOFF) {
        return x - ln2;
    } else {
        return std::log(std::cosh(x));
    }
}

/**
 * @brief Calculates lncosh for a complex value.
 *
 * @param x val
 *
 * @return ln(cosh(x))
 */
static inline std::complex<double> lncosh(std::complex<double> x) {
    const double xr = x.real();
    const double xi = x.imag();

    // Calculate the real lncosh
    std::complex<double> ret = lncosh(xr);

    // Calculate the complete lncosh with tanh, sin, which do not go to
    // infinity with real values.
    ret += std::log(
        std::complex<double>(std::cos(xi), std::tanh(xr) * std::sin(xi)));

    return ret;
}

/**
 * @brief A `MatrixXcd` wrapper for lncosh, applying lncosh to all elements.
 *
 * @param x The reference to the matrix.
 *
 * @return MatrixXcd result.
 */
static inline Eigen::MatrixXcd lncosh(const Eigen::MatrixXcd& x) {
    Eigen::MatrixXcd ret(x.rows(), x.cols());
    for (size_t i = 0; i < static_cast<size_t>(ret.size()); i++) {
        ret(i) = lncosh(x(i));
    }
    return ret;
}

static inline Eigen::MatrixXcd cosh2(const Eigen::MatrixXcd& x) {
    return (1 + x.real().array().pow(2) / 2) * x.imag().array().cos() +
           std::complex<double>(0, 1) * x.real().array() *
               x.imag().array().sin();
}

static inline Eigen::MatrixXcd tanh2(const Eigen::MatrixXcd& x) {
    return (x.real().array() * x.imag().array().cos() +
            std::complex<double>(0, 1) * (1 + x.real().array().pow(2) / 2) *
                x.imag().array().sin()) /
           cosh2(x).array();
}

static inline Eigen::MatrixXcd cosh1(const Eigen::MatrixXcd& x) {
    return x.array().cosh();
}

static inline Eigen::MatrixXcd tanh1(const Eigen::MatrixXcd& x) {
    Eigen::MatrixXcd r = x.array().tanh();
    return r.array().isFinite().select(r, 0.);
}

}  // namespace math
