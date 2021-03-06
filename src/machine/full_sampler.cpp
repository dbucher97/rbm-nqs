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

#include <gmp.h>
#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
//
#include <machine/full_sampler.hpp>
#include <machine/rbm.hpp>

using namespace machine;

full_sampler::full_sampler(rbm& rbm_, size_t bp)
    : Base{rbm_}, bits_parallel_{bp} {}

void full_sampler::sample(size_t) {
    for (auto agg : aggs_) {
        agg->set_zero();
    }
    size_t b_len = (size_t)std::pow(2, bits_parallel_);
    mpz_t max;
    mpz_init(max);
    mpz_set_ui(max, 1);
    mpz_mul_2exp(max, max, rbm_.n_visible);

    double p_total = 0;

#pragma omp parallel for
    for (size_t b = 0; b < b_len; b++) {
        mpz_t n, n_last;
        mpz_init2(n, rbm_.n_visible);
        mpz_init2(n_last, rbm_.n_visible);
        mpz_set_ui(n, b);

        Eigen::MatrixXcd state(rbm_.n_visible, 1);
        get_state(n, state);
        Eigen::MatrixXcd thetas = rbm_.get_thetas(state);

        std::vector<size_t> flips;

        while (mpz_cmp(n, max) < 0) {
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

            mpz_set(n_last, n);
            mpz_add_ui(n, n, b_len);
            mpz_xor(n_last, n, n_last);
            get_flips(n_last, flips, state);
            rbm_.update_thetas(state, flips, thetas);
            for (auto f : flips) {
                state(f) *= -1;
            }
        }

        mpz_clear(n);
        mpz_clear(n_last);
    }
    mpz_clear(max);
    for (auto agg : aggs_) {
        agg->finalize(p_total);
    }
}

void full_sampler::get_state(mpz_t& n, Eigen::MatrixXcd& state) {
    state.setConstant(1);
    for (size_t i = 0; i < rbm_.n_visible; i++) {
        if (mpz_tstbit(n, i)) state(i) = -1;
    }
}

void full_sampler::get_flips(mpz_t& x, std::vector<size_t>& flips,
                             Eigen::MatrixXcd& state) {
    flips.clear();
    for (size_t i = 0; i < rbm_.n_visible; i++) {
        if (mpz_tstbit(x, i)) {
            flips.push_back(i);
        }
    }
}
