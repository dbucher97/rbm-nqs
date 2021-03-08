/**
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
#include <omp.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <random>
//
#include <machine/metropolis_sampler.hpp>
#include <operators/base_op.hpp>

using namespace machine;

metropolis_sampler::metropolis_sampler(rbm_base& rbm, std::mt19937& rng,
                                       size_t n_chains, size_t step_size,
                                       size_t warmup_steps)
    : Base{rbm},
      rng_{rng},
      n_chains_{n_chains},
      step_size_{step_size},
      warmup_steps_{warmup_steps},
      f_dist_{0, rbm.n_visible - 1} {}

void metropolis_sampler::sample(size_t total_samples) {
    // initialize aggs
    for (auto agg : aggs_) {
        agg->set_zero();
    }
    size_t samples_per_chain = total_samples / n_chains_;
    size_t residue = total_samples - samples_per_chain * n_chains_;
#pragma omp parallel for
    for (size_t c = 0; c < n_chains_; c++) {
        sample_chain(samples_per_chain + (c == 0 ? residue : 0));
    }
    for (auto agg : aggs_) {
        agg->finalize(total_samples);
    }
}

void metropolis_sampler::sample_chain(size_t total_samples) {
    size_t total_steps = total_samples * step_size_ + warmup_steps_;

    // Initilaize state
    Eigen::MatrixXcd state(rbm_.n_visible, 1);
    for (size_t i = 0; i < rbm_.n_visible; i++) {
        state(i) = u_dist_(rng_) < 0.5 ? 1 : -1;
    }

    // Retrieve thetas for state
    Eigen::MatrixXcd thetas = rbm_.get_thetas(state);

    // Do the Metropolis sampling
    for (size_t step = 0; step < total_steps; step++) {
        std::vector<size_t> flips = {f_dist_(rng_)};
        if (rbm_.flips_accepted(u_dist_(rng_), state, flips, thetas)) {
            for (auto& flip : flips) state(flip) *= -1;
        }
        if ((step >= warmup_steps_) &&
            ((step - warmup_steps_) % step_size_ == 0)) {
            for (auto op : ops_) {
                op->evaluate(rbm_, state, thetas);
            }
            for (auto agg : aggs_) {
                agg->aggregate();
            }
        }
    }
}

