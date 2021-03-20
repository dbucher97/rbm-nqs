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

// #include <gmp.h>

#include <Eigen/Dense>
#include <random>
#include <vector>
//
#include <machine/full_sampler.hpp>
#include <machine/rbm_base.hpp>

namespace machine {

class exact_sampler : public full_sampler {
    using Base = full_sampler;

    std::mt19937 rng_;
    std::uniform_int_distribution<size_t> u_dist_;

    void get_flips(size_t, std::vector<size_t>&);

   public:
    exact_sampler(rbm_base&, std::mt19937&);

    virtual void sample(size_t) override;
};

}  // namespace machine
