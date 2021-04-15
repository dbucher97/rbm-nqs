/*
 * Copyright (C) 2021  David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <machine/correlator.hpp>

using namespace machine;

correlator::correlator(const std::vector<size_t>& idxs) : idxs_{idxs} {}

correlator::evaluate(Eigen::MatrixXcd& state) {
    std::complex<double> ret = 1;
    for (auto i : idxs) {
        ret *= state[i];
    }
    return ret;
}
