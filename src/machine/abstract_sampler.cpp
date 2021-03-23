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
#include <machine/abstract_sampler.hpp>
#include <operators/base_op.hpp>

using namespace machine;

abstract_sampler::abstract_sampler(rbm_base& rbm) : rbm_{rbm} {}

void abstract_sampler::register_ops(
    const std::vector<operators::base_op*>& ops) {
    // Push the ops into local vector
    ops_.reserve(ops.size());
    for (auto op : ops) {
        ops_.push_back(op);
    }
}

void abstract_sampler::register_op(operators::base_op* op_ptr) {
    register_ops({op_ptr});
}

void abstract_sampler::clear_ops() { ops_.clear(); }

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
