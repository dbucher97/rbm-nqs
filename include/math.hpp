/**
 * include/math.hpp
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

#include <Eigen/Dense>
#include <cmath>
#include <complex>

static const double ln2 = std::log(2);

static double lncosh(double x) {
    x = std::abs(x);
    if (x > 16.7) {
        return x - ln2;
    } else {
        return std::log(std::cosh(x));
    }
}

static std::complex<double> lncosh(std::complex<double> x) {
    const double xr = x.real();
    const double xi = x.imag();

    std::complex<double> ret = lncosh(xr);
    ret += std::log(
        std::complex<double>(std::cos(xi), std::tanh(xr) * std::sin(xi)));

    return ret;
}

static Eigen::MatrixXcd lncosh(const Eigen::MatrixXcd& x) {
    Eigen::MatrixXcd ret(x.rows(), x.cols());
    for (size_t i = 0; i < static_cast<size_t>(ret.size()); i++) {
        ret(i) = lncosh(x(i));
    }
    return ret;
}
