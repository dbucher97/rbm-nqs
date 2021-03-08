/**
 * src/machine/full_sampler.cpp
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

// #include <gmp.h>
#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
//
#include <machine/full_sampler.hpp>
#include <machine/rbm_base.hpp>

using namespace machine;

full_sampler::full_sampler(rbm_base& rbm_, size_t bp)
    : Base{rbm_}, bits_parallel_{bp} {}

void full_sampler::sample(size_t) {
    for (auto agg : aggs_) {
        agg->set_zero();
    }
    size_t b_len = (size_t)std::pow(2, bits_parallel_);
    size_t max = (size_t)std::pow(2, rbm_.n_visible - bits_parallel_);

    double p_total = 0;

#pragma omp parallel for
    for (size_t b = 0; b < b_len; b++) {
        size_t x = 0;
        size_t x_last = 0;
        size_t flip;

        Eigen::MatrixXcd state(rbm_.n_visible, 1);
        get_state(b, state);
        Eigen::MatrixXcd thetas = rbm_.get_thetas(state);

        for (size_t i = 1; i <= max; i++) {
            auto psi = rbm_.psi(state, thetas);
            double p = std::pow(std::abs(psi), 2);
#pragma omp critical
            { p_total += p; }
            for (auto op : ops_) {
                op->evaluate(rbm_, state, thetas);
            }
            for (auto agg : aggs_) {
                agg->aggregate(p);
            }
            if (i != max) {
                // gray_code
                x = i ^ (i >> 1);
                flip = std::log2l(x ^ x_last) + bits_parallel_;
                x_last = x;
                rbm_.update_thetas(state, {flip}, thetas);
                state(flip) *= -1;
            }
        }
    }
    for (auto agg : aggs_) {
        agg->finalize(p_total);
    }
}

void full_sampler::get_state(size_t n, Eigen::MatrixXcd& state) {
    state.setConstant(1);
    for (size_t i = 0; i < rbm_.n_visible; i++) {
        if (test_bit(n, i)) state(i) = -1;
    }
}

bool full_sampler::test_bit(size_t s, size_t i) { return (s >> i) & 1; }

