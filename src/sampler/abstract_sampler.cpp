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

#include <vector>
//
#include <operators/base_op.hpp>
#include <sampler/abstract_sampler.hpp>
#include <tools/time_keeper.hpp>

using namespace sampler;

abstract_sampler::abstract_sampler(machine::abstract_machine& rbm,
                                   size_t n_samples)
    : rbm_{rbm}, n_samples_{n_samples} {}

void abstract_sampler::register_ops(
    const std::vector<operators::base_op*>& ops) {
    // Push the ops into local vector
    for (auto op : ops) {
        register_op(op);
    }
}

void abstract_sampler::register_op(operators::base_op* op_ptr) {
    ops_.push_back(op_ptr);
}

void abstract_sampler::register_op(operators::local_op_chain* op_ptr) {
    chains_.push_back(op_ptr);
    register_ops(op_ptr->get_ops());
}

void abstract_sampler::clear_ops() {
    ops_.clear();
    chains_.clear();
}

void abstract_sampler::register_aggs(
    const std::vector<operators::aggregator*>& aggs) {
    // Push the aggs into local vector
    aggs_.reserve(aggs.size());
    for (auto agg : aggs) {
        aggs_.push_back(agg);
    }
}

void abstract_sampler::register_agg(operators::aggregator* agg_ptr) {
    register_aggs({agg_ptr});
}

void abstract_sampler::clear_aggs() { aggs_.clear(); }

size_t abstract_sampler::get_n_samples() const { return n_samples_; }
void abstract_sampler::set_n_samples(size_t samples) { n_samples_ = samples; }

void abstract_sampler::evaluate_and_aggregate(const Eigen::MatrixXcd& state,
                                              machine::rbm_context& context,
                                              double p) const {
    time_keeper::start("Evaluate");
    // Evaluate operators
    for (auto& op : ops_) {
        op->evaluate(rbm_, state, context);
    }

    for (auto& chain : chains_) {
        chain->finailize();
    }
    time_keeper::end("Evaluate");
    // Evaluate aggregators
    time_keeper::start("Aggregate");
    for (auto& agg : aggs_) {
        agg->aggregate(p);
    }
    time_keeper::end("Aggregate");
}
