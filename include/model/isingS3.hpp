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

extern Eigen::Matrix4cd ixx;    ///< 2 site Pauli-z operator
extern Eigen::Matrix4cd iyy;    ///< 2 site Pauli-z operator
extern Eigen::Matrix4cd izz;    ///< 2 site Pauli-z operator
extern Eigen::Matrix4cd ix_yz;  ///< 2 site Pauli-x operator in y z basis
extern Eigen::Matrix4cd ix_zy;  ///< 2 site Pauli-x operator in z y basis
extern Eigen::Matrix4cd iy_xz;  ///< 2 site Pauli-y operator in x z basis
extern Eigen::Matrix4cd iy_zx;  ///< 2 site Pauli-y operator in z x basis
extern Eigen::Matrix4cd iz_xy;  ///< 2 site Pauli-z operator in x y basis
extern Eigen::Matrix4cd iz_yx;  ///< 2 site Pauli-z operator in y x basis
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
};
}  // namespace model
