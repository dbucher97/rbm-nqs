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

#include <machine/context.hpp>

using namespace machine;

pfaff_context::pfaff_context(const pfaff_context& other)
    : inv{other.inv},
      pfaff{other.pfaff},
      update_factor{other.update_factor},
      exp{other.exp} {
}

pfaff_context::pfaff_context(pfaff_context&& other) noexcept
    : inv{std::move(other.inv)},
      pfaff{std::move(other.pfaff)},
      update_factor{std::move(other.update_factor)},
      exp{std::move(other.exp)} {
}

pfaff_context& pfaff_context::operator=(const pfaff_context& other) {
    inv = other.inv;
    pfaff = other.pfaff;
    update_factor = other.update_factor;
    exp = other.exp;
    return *this;
}

pfaff_context& pfaff_context::operator=(pfaff_context&& other) {
    inv = std::move(other.inv);
    pfaff = std::move(other.pfaff);
    update_factor = std::move(other.update_factor);
    exp = std::move(other.exp);
    return *this;
}

rbm_context::rbm_context(const Eigen::MatrixXcd& thetas) : thetas{thetas} {}

rbm_context::rbm_context(const Eigen::MatrixXcd& thetas,
                         const pfaff_context& other)
    : thetas{thetas}, pfaff_{std::make_unique<pfaff_context>(other)} {}

rbm_context::rbm_context(const Eigen::MatrixXcd& thetas, pfaff_context&& other)
    : thetas{thetas},
      pfaff_{
          std::make_unique<pfaff_context>(std::forward<pfaff_context>(other))} {
}

rbm_context::rbm_context(const rbm_context& other) {
    thetas = other.thetas;
    if (other.pfaff_) pfaff_ = std::make_unique<pfaff_context>(*other.pfaff_);
}

rbm_context& rbm_context::operator=(rbm_context& other) {
    std::swap(thetas, other.thetas);
    std::swap(pfaff_, other.pfaff_);
    return *this;
}

pfaff_context& rbm_context::pfaff() {
    if (!pfaff_) pfaff_ = std::make_unique<pfaff_context>();
    return *pfaff_;
}

const pfaff_context& rbm_context::pfaff() const { return *pfaff_; }
