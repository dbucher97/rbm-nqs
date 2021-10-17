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
#include <machine/spin_state.hpp>
#include <operators/base_op.hpp>
#include <sampler/abstract_sampler.hpp>
#include <tools/time_keeper.hpp>

#include "mpi.h"

using namespace sampler;

abstract_sampler::abstract_sampler(machine::abstract_machine& rbm,
                                   size_t n_samples, int pfaff_refresh,
                                   int lut_exchange)
    : rbm_{rbm},
      n_samples_{n_samples},
      pfaff_refresh_(pfaff_refresh),
      lut_exchange_{lut_exchange} {}

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

void abstract_sampler::evaluate_and_aggregate(const machine::spin_state& state,
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

bool abstract_sampler::pfaffian_refresh(
    const machine::spin_state& state, machine::pfaff_context& context, int i,
    const std::vector<size_t>& flips) const {
    if (rbm_.has_pfaffian() && pfaff_refresh_ && i % pfaff_refresh_ == 0) {
        machine::spin_state state2 = state;
        state2.flip(flips);
        context = rbm_.get_pfaffian().get_context(state2);
        return true;
    }
    return false;
}

void abstract_sampler::exchange_luts(int i) const {
    if (lut_exchange_ && i > 0 && i % lut_exchange_ == 0) {
        time_keeper::start("LUT Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
        time_keeper::end("LUT Barrier");
        time_keeper::start("LUT Exchange");
        rbm_.exchange_luts();
        time_keeper::end("LUT Exchange");
    }
}
