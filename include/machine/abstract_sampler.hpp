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

#pragma once

#include <Eigen/Dense>
#include <complex>
#include <random>
#include <vector>
//
#include <machine/rbm_base.hpp>
#include <operators/aggregator.hpp>
#include <operators/base_op.hpp>

namespace machine {

class abstract_sampler {
   protected:
    rbm_base& rbm_;

    std::vector<operators::base_op*> ops_;
    std::vector<operators::aggregator*> aggs_;

   public:
    abstract_sampler(rbm_base&);
    virtual ~abstract_sampler() = default;

    virtual void sample(size_t) = 0;

    void register_ops(const std::vector<operators::base_op*>&);
    void register_op(operators::base_op*);
    void clear_ops();
    void register_aggs(const std::vector<operators::aggregator*>&);
    void register_agg(operators::aggregator*);
    void clear_aggs();
};

}  // namespace machine
