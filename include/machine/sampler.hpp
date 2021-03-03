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
#pragma once

#include <Eigen/Dense>
#include <complex>
#include <random>
#include <vector>
//
#include <machine/rbm.hpp>
#include <operators/aggregator.hpp>
#include <operators/base_op.hpp>

namespace machine {

class sampler {
   public:
    sampler(rbm&, std::mt19937&);

    void sample(size_t, size_t = 5, size_t = 20, bool = false);

    void register_ops(const std::vector<operators::base_op*>&);
    void register_op(operators::base_op*);
    void register_aggs(const std::vector<operators::aggregator*>&);
    void register_agg(operators::aggregator*);

   private:
    rbm& rbm_;
    std::mt19937& rng_;

    Eigen::MatrixXcd state_;
    Eigen::MatrixXcd thetas_;

    std::uniform_int_distribution<size_t> f_dist_;

    std::vector<operators::base_op*> ops_;
    std::vector<operators::aggregator*> aggs_;

    size_t current_step_ = 0;
    size_t current_sample_ = 0;

    std::uniform_real_distribution<double> u_dist_{0, 1};

    void step();
};

}  // namespace machine
