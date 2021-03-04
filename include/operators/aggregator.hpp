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
//
#include <operators/base_op.hpp>

namespace operators {

class aggregator {
    bool real_;

   protected:
    const base_op& op_;
    Eigen::MatrixXcd result_;

    virtual Eigen::MatrixXcd aggregate_();

    aggregator(const base_op&, size_t, size_t, bool = false);

   public:
    aggregator(const base_op&, bool = false);
    virtual ~aggregator() = default;

    void aggregate(double = 1);

    Eigen::MatrixXcd& get_result();
    void finalize(double);

    void set_zero() { result_.setZero(); }
};

class prod_aggregator : public aggregator {
    using Base = aggregator;

    const base_op& scalar_;

    virtual Eigen::MatrixXcd aggregate_() override;

   public:
    prod_aggregator(const base_op&, const base_op&);
};

class outer_aggregator : public aggregator {
    using Base = aggregator;

    virtual Eigen::MatrixXcd aggregate_() override;

   public:
    outer_aggregator(const base_op&);
};

}  // namespace operators
