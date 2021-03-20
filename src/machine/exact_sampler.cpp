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

// #include <gmp.h>
#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <set>
//
#include <machine/exact_sampler.hpp>
#include <machine/rbm_base.hpp>

using namespace machine;

exact_sampler::exact_sampler(rbm_base& rbm_, std::mt19937& rng)
    : Base{rbm_, 0},
      rng_{rng},
      u_dist_{0, static_cast<size_t>(1 << rbm_.n_visible) - 1} {}

void exact_sampler::sample(size_t samples) {
    for (auto agg : aggs_) {
        agg->set_zero();
    }
    std::set<size_t> states_set;
    samples = std::min(static_cast<size_t>(1 << rbm_.n_visible), samples);
    while (states_set.size() != samples) {
        states_set.insert(u_dist_(rng_));
    }
    std::vector<size_t> states{states_set.begin(), states_set.end()};

    double p_total = 0;

    size_t samples_per_chain = samples / omp_get_num_threads();

#pragma omp parallel for
    for (size_t i = 0; i < omp_get_num_threads(); i++) {
        size_t max = samples_per_chain * (i + 1);
        if (i + 1 == omp_get_num_threads()) max = samples;

        Eigen::MatrixXcd state(rbm_.n_visible, 1);
        get_state(states[samples_per_chain * i], state);
        Eigen::MatrixXcd thetas = rbm_.get_thetas(state);
        std::vector<size_t> flips;

        for (size_t j = samples_per_chain * i; j < max; j++) {
            auto psi = rbm_.psi(state, thetas);
            double p = std::pow(std::abs(psi), 2);
#pragma omp critical
            { p_total += p; }
            for (auto& op : ops_) {
                op->evaluate(rbm_, state, thetas);
            }
            for (auto& agg : aggs_) {
                agg->aggregate(p);
            }
            if (j != max) {
                get_flips(states[i] ^ states[i + 1], flips);
                rbm_.update_thetas(state, flips, thetas);
                for (auto& flip : flips) state(flip) *= -1;
            }
        }
    }

    for (auto agg : aggs_) {
        agg->finalize(p_total);
    }
}

void exact_sampler::get_flips(size_t v, std::vector<size_t>& flips) {
    flips.clear();
    for (size_t i = 0; i < rbm_.n_visible; i++) {
        if (test_bit(v, i)) flips.push_back(i);
    }
}
