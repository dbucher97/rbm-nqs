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

#define CHUNK_SIZE 32

#include <omp.h>

#include <iostream>
//
#include <math.hpp>
#include <tools/mpi.hpp>

using namespace math;

std::complex<double> lncosh_(const std::complex<double>* xd, const size_t start, const size_t end) {
    double reabs, lncoshre, sinim, cosim, gretre = 0, gretim = 0;
    for (size_t i = start; i < end; i++) {
            reabs = std::abs(std::real(xd[i]));

            lncoshre = reabs - M_LN2;
            reabs = std::exp(-2. * reabs);

            lncoshre += LOG1P(reabs);
            sinim = std::sin(std::imag(xd[i]));
            cosim = std::cos(std::imag(xd[i]));
            sinim *=
                std::copysign((1. - reabs) / (1. + reabs), std::real(xd[i]));

            lncoshre +=
                0.5 * std::log(std::pow(sinim, 2.) + std::pow(cosim, 2.));
            gretre += lncoshre;
            gretim += std::atan2(sinim, cosim);
    }
    return {gretre, gretim};
}
 
std::complex<double> math::lncosh(const Eigen::MatrixXcd& x) {
    const std::complex<double>* xd = x.data();
    double reres = 0;
    double imres = 0;

    const size_t n_chunks = omp_get_max_threads();
    const size_t chunk_size = x.size() / n_chunks;

#pragma omp parallel for reduction(+:reres) reduction(+:imres)
    for (size_t j = 0; j < n_chunks; j++) {
        std::complex<double> lres;
        const size_t end = ((j + 1) == n_chunks) ? x.size() : (j + 1) * chunk_size; 

        lres = lncosh_(xd, j * chunk_size, end); 

        reres += std::real(lres);
        imres += std::imag(lres);
    }


    //}
    return {reres, imres};
}

// std::complex<double> math::lncosh(const Eigen::MatrixXcd& x) {
//     return lncosh_(x.data(), 0, x.size());
// }
