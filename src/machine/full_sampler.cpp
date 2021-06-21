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
#include <fstream>
//
#include <machine/full_sampler.hpp>
#include <machine/rbm_base.hpp>
#include <tools/ini.hpp>
#include <tools/state.hpp>

using namespace machine;

full_sampler::full_sampler(rbm_base& rbm_, size_t bp)
    : Base{rbm_, (size_t)(1 << rbm_.n_visible)}, bits_parallel_{bp} {}

void full_sampler::sample(bool keep_state) {
    // Initialize aggregators
    for (auto agg : aggs_) {
        agg->set_zero();
    }
    // Number of parallel gray code runs
    size_t b_len = (size_t)std::pow(2, bits_parallel_);

    // Number of total pit flips
    size_t max = (size_t)std::pow(2, rbm_.n_visible - bits_parallel_);

    double p_total = 0;

    // The state vector if state should be kept
    Eigen::MatrixXcd vec(static_cast<size_t>(1 << rbm_.n_visible), 1);

    // Start the parallel runs
#pragma omp parallel for
    for (size_t b = 0; b < b_len; b++) {
        size_t x = 0;
        size_t x_last = 0;
        size_t flip;

        // Get the state for `b`
        Eigen::MatrixXcd state(rbm_.n_visible, 1);
        tools::num_to_state(b, state);

        // Precalculate context
        auto context = rbm_.get_context(state);

        // Do the spin flips according to gray codes and evalueate observables
        for (size_t i = 1; i <= max; i++) {
            // Get the \psi of the current state and calculate probability
            auto psi = rbm_.psi(state, context);
            double p = std::pow(std::abs(psi), 2);

            if (p > 1e-20) {
                // If keep state store \psi into the state vector
                if (keep_state) {
#pragma omp critical
                    vec(tools::state_to_num(state)) = psi;
                }

                // Cumulate probability for normalization
#pragma omp critical
                p_total += p;

                // Evaluate operators
                for (auto op : ops_) {
                    op->evaluate(rbm_, state, context);
                }
                // Evaluate aggregators
                for (auto agg : aggs_) {
                    agg->aggregate(p);
                }
            }

            // Do the gray code update
            if (i != max) {
                // Gray code update
                x = i ^ (i >> 1);

                // Calculate the bit which needs to be flipped
                flip = std::log2l(x ^ x_last) + bits_parallel_;
                x_last = x;

                // Update \thetas and state
                rbm_.update_context(state, {flip}, context);
                state(flip) *= -1;
            }
        }
    }
    // Print the state vector if `keep_state`
    if (keep_state) {
        std::ofstream statefile{ini::name + ".state"};
        statefile << vec;
        statefile.close();
    }
    for (auto agg : aggs_) {
        agg->finalize(p_total);
    }
}
