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
#include <vector>
//
#include <lattice/honeycomb.hpp>
#include <operators/base_op.hpp>
#include <operators/bond_op.hpp>

namespace model {
extern Eigen::Matrix4cd sxsx;
extern Eigen::Matrix4cd sysy;
extern Eigen::Matrix4cd szsz;

class kitaev {
   public:
    kitaev(size_t size, double J) : kitaev(size, {J, J, J}) {}
    kitaev(size_t size, const std::array<double, 3>& J)
        : lat{size},
          // hamiltonian{lat.get_bonds(),
          //             {J[0] * (sysy + szsz), J[1] * (sxsx + szsz),
          //              J[2] * (sxsx + sysy)}} {}
          hamiltonian{lat.get_bonds(),
                      {J[0] * sxsx, J[1] * sysy, J[2] * szsz}} {}

    operators::base_op& get_hamiltonian() { return hamiltonian; }
    lattice::bravais& get_lattice() { return lat; };

   private:
    lattice::honeycomb lat;
    operators::bond_op hamiltonian;
};
}  // namespace model
