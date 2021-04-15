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
#include <omp.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <random>
//
#include <machine/metropolis_sampler.hpp>
#include <operators/base_op.hpp>
#include <tools/logger.hpp>

using namespace machine;

metropolis_sampler::metropolis_sampler(rbm_base& rbm, std::mt19937& rng,
                                       size_t n_chains, size_t step_size,
                                       size_t warmup_steps)
    : Base{rbm},
      rng_{rng},
      n_chains_{n_chains},
      step_size_{step_size},
      warmup_steps_{warmup_steps},
      single_flip_prob_{1.},
      f_dist_{0, rbm.n_visible - 1} {}

void metropolis_sampler::sample(size_t total_samples) {
    // Initialize aggregators
    for (auto agg : aggs_) {
        agg->set_zero();
    }

    // Divide the `total_samples` between the chains.
    size_t samples_per_chain = total_samples / n_chains_;
    size_t residue = total_samples - samples_per_chain * n_chains_;

    // Initialize acceptance_rate
    acceptance_rate_ = 0;
    single_flip_prob_ *= 0.996;

    // Run the chains in parallel.
#pragma omp parallel for
    for (size_t c = 0; c < n_chains_; c++) {
        double ar = sample_chain(samples_per_chain + (c == 0 ? residue : 0));
        // Accumulate acceptance_rate
#pragma omp critical
        acceptance_rate_ += ar;
    }
    // Average acceptance rate.
    acceptance_rate_ /= n_chains_;

    // Finalize aggregators
    for (auto agg : aggs_) {
        agg->finalize(total_samples);
    }
}

double metropolis_sampler::sample_chain(size_t total_samples) {
    size_t total_steps = total_samples * step_size_ + warmup_steps_;
    size_t ar = 0;

    // Initilaize random state
    Eigen::MatrixXcd state(rbm_.n_visible, 1);
    size_t ups = 0;
    for (size_t i = 0; i < rbm_.n_visible; i++) {
        state(i) = u_dist_(rng_) < 0.5 ? 1. : -1.;
        ups += (state(i) == 1.);
    }
    // if (ups % 2 == 1) {
    //     state(0) *= -1;
    // }

    auto bonds = rbm_.get_lattice().get_bonds();
    std::uniform_int_distribution<size_t> b_dist(0, bonds.size() - 1);

    // Retrieve thetas for state
    Eigen::MatrixXcd thetas = rbm_.get_thetas(state);

    std::vector<size_t> flips;

    // Do the Metropolis sampling
    for (size_t step = 0; step < total_steps; step++) {
        // Get the flips vector by randomly selecting one site.
        flips.clear();
        // With probability 1/2 flip a second site. This scheme needs further
        // analysis
        flips.push_back(f_dist_(rng_));
        // if (u_dist_(rng_) < 0.5) {
        //     size_t flips2 = f_dist_(rng_);
        //     while (flips[0] == flips2) {
        //         flips2 = f_dist_(rng_);
        //     }
        //     flips.push_back(flips2);
        // }
        // auto& b = bonds[b_dist(rng_)];
        // std::vector<size_t> flips = {b.a, b.b};
        // if (u_dist_(rng_) < 0.5) {
        //     flips.pop_back();
        // }

        // try to accept flips, if flips are accepted, thetas are automatically
        // updated and state needs updating.
#ifndef ALT_POP
        if (rbm_.flips_accepted(u_dist_(rng_), state, flips, thetas)) {
#else
        if (rbm_.flips_accepted_alt(u_dist_(rng_), state, flips, thetas)) {
#endif
            ar++;
            for (auto& flip : flips) state(flip) *= -1;
        }

        // If a sample is required
        if ((step >= warmup_steps_) &&
            ((step - warmup_steps_) % step_size_ == 0)) {
            // Evaluate oprators
            for (auto op : ops_) {
                op->evaluate(rbm_, state, thetas);
            }
            // Evaluate aggregators
            for (auto agg : aggs_) {
                agg->aggregate();
            }
            // thetas = rbm_.get_thetas(state);
        }
    }

    // Normalize acceptance rate
    return ar / (double)total_steps;
}

void metropolis_sampler::log() {
    logger::log(acceptance_rate_, "AccetpanceRate");
}
