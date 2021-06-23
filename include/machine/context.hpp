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
#include <memory>

namespace machine {

struct pfaff_context {
    Eigen::MatrixXcd inv;
    std::complex<double> pfaff;
    int exp;

    pfaff_context() {}

    pfaff_context(const pfaff_context &other) {
        inv = other.inv;
        pfaff = other.pfaff;
    }

    pfaff_context& operator=(const pfaff_context& other) {
        inv = other.inv;
        pfaff = other.pfaff;
        return *this;
    }

    pfaff_context& operator=(pfaff_context&& other) {
        inv = std::move(other.inv);
        pfaff = std::move(other.pfaff);
        return *this;
    }
};

struct rbm_context {
    Eigen::MatrixXcd thetas;

    rbm_context() {}
    rbm_context(const Eigen::MatrixXcd& thetas) : thetas{thetas} {}

    rbm_context(const rbm_context &other) {
        thetas = other.thetas;
        if (other.pfaff_)
            pfaff_ = std::make_unique<pfaff_context>(*other.pfaff_);
    }

    rbm_context &operator=(rbm_context &other) {
        std::swap(thetas, other.thetas);
        std::swap(pfaff_, other.pfaff_);
        return *this;
    }

    pfaff_context &pfaff() {
        if (!pfaff_) pfaff_ = std::make_unique<pfaff_context>();
        return *pfaff_;
    }

   private:
    std::unique_ptr<pfaff_context> pfaff_;
};

}  // namespace machine
