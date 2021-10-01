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
#include <iostream>
#include <memory>
#include <unordered_map>

namespace machine {

struct pfaff_context {
    Eigen::MatrixXcd inv;
    std::complex<double> pfaff;
    std::complex<double> update_factor;
    int exp;

    pfaff_context() {}

    pfaff_context(const pfaff_context& other);

    pfaff_context(pfaff_context&& other) noexcept;

    pfaff_context& operator=(const pfaff_context& other);

    pfaff_context& operator=(pfaff_context&& other);
};

struct rbm_context {
    Eigen::MatrixXcd thetas;

   public:
    rbm_context() {}

    rbm_context(const Eigen::MatrixXcd& thetas);
    rbm_context(const Eigen::MatrixXcd& thetas, const pfaff_context& other);
    rbm_context(const Eigen::MatrixXcd& thetas, pfaff_context&& other);

    rbm_context(const rbm_context& other);

    rbm_context& operator=(rbm_context& other);
    rbm_context& operator=(const rbm_context& other);

    pfaff_context& pfaff();

    const pfaff_context& pfaff() const;

   private:
    std::unique_ptr<pfaff_context> pfaff_;
};

}  // namespace machine
