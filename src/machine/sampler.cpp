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

#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <random>
//
#include <machine/sampler.hpp>
#include <operators/base_op.hpp>

using namespace machine;

sampler::sampler(rbm& ma, std::mt19937& rng)
    : rbm_{ma},
      rng_{rng},
      state_(ma.n_visible, 1),
      thetas_(ma.n_alpha, ma.symmetry_size()),
      f_dist_{0, ma.n_visible - 1} {}

void sampler::sample(size_t total_samples, size_t steps_per_sample,
                     size_t warmup_steps, bool print) {
    current_sample_ = 0;
    size_t total_steps = total_samples * steps_per_sample + warmup_steps;
    for (size_t i = 0; i < rbm_.n_visible; i++) {
        state_(i) = u_dist_(rng_) < 0.5 ? 1 : -1;
    }
    for (auto agg : aggs_) {
        agg->set_zero();
    }
    thetas_ = rbm_.get_thetas(state_);

    for (current_step_ = 0; current_step_ < total_steps; current_step_++) {
        step();
        if ((current_step_ >= warmup_steps) &&
            ((current_step_ - warmup_steps) % steps_per_sample == 0)) {
            // std::cout << state_.transpose() << std::endl;
            if (print) std::cout << state_ << std::endl;
            for (auto op : ops_) {
                op->evaluate(rbm_, state_, thetas_);
            }
            for (auto agg : aggs_) {
                agg->aggregate();
            }
        }
    }
    // std::cout << std::endl;
}

void sampler::step() {
    std::vector<size_t> flips = {f_dist_(rng_)};
    if (rbm_.flips_accepted(u_dist_(rng_), state_, flips, thetas_)) {
        for (auto& flip : flips) state_(flip) *= -1;
    }
}

void sampler::register_ops(const std::vector<operators::base_op*>& ops) {
    ops_.reserve(ops.size());
    for (auto op : ops) {
        ops_.push_back(op);
    }
}

void sampler::register_op(operators::base_op* op_ptr) {
    register_ops({op_ptr});
}

void sampler::register_aggs(const std::vector<operators::aggregator*>& aggs) {
    aggs_.reserve(aggs.size());
    for (auto agg : aggs) {
        aggs_.push_back(agg);
    }
}

void sampler::register_agg(operators::aggregator* agg_ptr) {
    register_aggs({agg_ptr});
}
