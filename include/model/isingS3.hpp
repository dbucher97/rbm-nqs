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

#include <array>
#include <complex>
#include <stdexcept>
#include <vector>
//
#include <lattice/honeycombS3.hpp>
#include <model/abstract_model.hpp>

/**
 * @brief Namespace for the physical models.
 */
namespace model {

/**
 * @brief The Honeycomb Kitaev Model.
 */
class isingS3 : public abstract_model {
    using Base = abstract_model;

   public:
    /**
     * @brief Ising model consturctor.
     *
     * @param size Number of unitcells in one direction.
     * @param J the coupling in all three directions
     */
    isingS3(size_t size, double J);

    /**
     * @brief Ising model consturctor.
     *
     * @param size Number of unitcells in one direction.
     * @param J the coupling in vector form
     */
    isingS3(size_t size, const std::vector<double>& J) : isingS3{size, J[0]} {}
};
}  // namespace model
