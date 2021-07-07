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
#include <fstream>
#include <string>
//
#include <operators/base_op.hpp>

namespace operators {
/**
 * @brief This operator does not evalueate something, but it plugs into the
 * operator interface to store the states to a file. It will be a list of all
 * the states sampled.
 */
class store_state : public base_op {
    using Base = base_op;
    std::ofstream file_;  ///< Output filestream.

   public:
    /**
     * @brief Store State constructor
     *
     * @param filename Name of the file to store the states to.
     */
    store_state(const std::string& filename);

    void evaluate(machine::abstract_machine&, const Eigen::MatrixXcd&,
                  const machine::rbm_context&) final;

    /**
     * @brief Destructor closes output stream.
     */
    ~store_state() { file_.close(); }
};
}  // namespace operators
