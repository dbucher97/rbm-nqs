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
#include <machine/abstract_sampler.hpp>
#include <machine/rbm_base.hpp>
#include <operators/aggregator.hpp>
#include <operators/base_op.hpp>

namespace machine {

class metropolis_sampler : public abstract_sampler {
    using Base = abstract_sampler;

   public:
    metropolis_sampler(rbm_base&, std::mt19937&, size_t = 1, size_t = 5,
                       size_t = 100);

    virtual void sample(size_t) override;

    size_t get_step_size() const;
    size_t get_warmup_steps() const;

    void set_step_size(size_t);
    void set_warmup_steps(size_t);

   private:
    std::mt19937& rng_;

    size_t n_chains_, step_size_, warmup_steps_;

    std::uniform_int_distribution<size_t> f_dist_;

    std::uniform_real_distribution<double> u_dist_{0, 1};

    void sample_chain(size_t);
};

}  // namespace machine
