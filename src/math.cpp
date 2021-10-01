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

#include <iostream>
//
#include <math.hpp>

using namespace math;

std::complex<double> math::lncosh(const Eigen::MatrixXcd& x) {
    double reabs, lncoshre, sinim, cosim;
    const std::complex<double>* xd = x.data();
    std::complex<double> ret = 0;

    /*int n_chunks = 1; //omp_get_max_threads();
    int chunk_size = x.size(); // / n_chunks;

// #pragma omp parallel for
    for (int j = 0; j < n_chunks; j++) {
        int end = (j + 1) * chunk_size;
        if(j == n_chunks - 1)
            end = x.size(); */
    for (int i = 0; i < x.size(); i++) {
        reabs = std::abs(std::real(xd[i]));

        lncoshre = reabs - M_LN2;
        reabs = std::exp(-2. * reabs);

        lncoshre += LOG1P(reabs);
        sinim = std::sin(std::imag(xd[i]));
        cosim = std::cos(std::imag(xd[i]));
        sinim *= std::copysign((1. - reabs) / (1. + reabs), std::real(xd[i]));

        lncoshre += 0.5 * std::log(std::pow(sinim, 2.) + std::pow(cosim, 2.));
        ret += std::complex<double>(lncoshre, std::atan2(sinim, cosim));
    }

    //}
    return ret;
}
