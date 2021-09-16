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

// log(1 + x) is faster than log1p in the intel case.
#ifdef __INTEL_COMPILER
#define LOG1P(x) std::log(1. + x)
#else
#define LOG1P(x) std::log1p(x)
#endif

#include <omp.h>
//
#include <math.hpp>

using namespace math;

void math::lncosh(const Eigen::MatrixXcd& x, Eigen::ArrayXXcd& res) {
    double reabs, lncoshre, sinim, cosim;
#pragma omp parallel for private(reabs, lncoshre, sinim, cosim)
    for (int i = 0; i < x.size(); i++) {
        reabs = std::abs(std::real(x(i)));

        lncoshre = reabs - M_LN2;
        reabs = std::exp(-2. * reabs);

        lncoshre += LOG1P(reabs);

        sinim = std::sin(std::imag(x(i)));
        cosim = std::cos(std::imag(x(i)));
        sinim *= std::copysign((1 - reabs) / (1 + reabs), std::real(x(i)));

        res(i) = std::complex<double>(
            lncoshre +
                0.5 * std::log(std::pow(cosim, 2.) + std::pow(sinim, 2.)),
            std::atan2(sinim, cosim));
    }
}
