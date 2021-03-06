/**
 * include/machine/full_sampler.hpp
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

#include <gmp.h>

#include <Eigen/Dense>
//
#include <machine/abstract_sampler.hpp>
#include <machine/rbm.hpp>

namespace machine {

class full_sampler : public abstract_sampler {
    using Base = abstract_sampler;

    size_t bits_parallel_;

    void get_state(mpz_t&, Eigen::MatrixXcd&);
    void get_flips(mpz_t&, std::vector<size_t>&, Eigen::MatrixXcd&);

   public:
    full_sampler(rbm&, size_t);

    virtual void sample(size_t = 0) override;
};

}  // namespace machine
