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

#include <cmath>
#include <machine/context.hpp>
//
#include <math.hpp>
#include <tools/mpi.hpp>

using namespace machine;

pfaff_context::pfaff_context(const pfaff_context& other)
    : inv{other.inv},
      pfaff{other.pfaff},
      update_factor{other.update_factor},
      exp{other.exp} {}

pfaff_context::pfaff_context(pfaff_context&& other) noexcept
    : inv{std::move(other.inv)},
      pfaff{std::move(other.pfaff)},
      update_factor{std::move(other.update_factor)},
      exp{std::move(other.exp)} {}

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

rbm_context::rbm_context(const Eigen::MatrixXcd& thetas, size_t cosh_mode)
    : thetas{thetas},
      coshthetas_(thetas.rows(), thetas.cols()),
      lncoshthetas_(thetas.rows(), thetas.cols()),
      cosh_mode_{cosh_mode} {
    init_cosh_funcs();
};

rbm_context::rbm_context(const Eigen::MatrixXcd& thetas,
                         const pfaff_context& other, size_t cosh_mode)
    : rbm_context{thetas, cosh_mode} {
    pfaff_ = std::make_unique<pfaff_context>(other);
}

rbm_context::rbm_context(const Eigen::MatrixXcd& thetas, pfaff_context&& other,
                         size_t cosh_mode)
    : rbm_context{thetas, cosh_mode} {
    pfaff_ =
        std::make_unique<pfaff_context>(std::forward<pfaff_context>(other));
}

rbm_context::rbm_context(const rbm_context& other) {
    thetas = other.thetas;
    if (other.pfaff_) pfaff_ = std::make_unique<pfaff_context>(*other.pfaff_);
    lncoshthetas_ = other.lncoshthetas_;
    coshthetas_ = other.coshthetas_;
    did_lncoshthetas_ = other.did_lncoshthetas_;
    did_coshthetas_ = other.did_coshthetas_;
    cosh_mode_ = other.cosh_mode_;
    init_cosh_funcs();
}

rbm_context& rbm_context::operator=(rbm_context& other) {
    std::swap(thetas, other.thetas);
    std::swap(pfaff_, other.pfaff_);
    std::swap(lncoshthetas_, other.lncoshthetas_);
    std::swap(coshthetas_, other.coshthetas_);
    std::swap(did_lncoshthetas_, other.did_lncoshthetas_);
    std::swap(did_coshthetas_, other.did_coshthetas_);
    std::swap(cosh_mode_, other.cosh_mode_);
    std::swap(cosh_, other.cosh_);
    std::swap(lncosh_, other.lncosh_);
    return *this;
}

rbm_context& rbm_context::operator=(const rbm_context& other) {
    thetas = other.thetas;
    if (other.pfaff_) pfaff_ = std::make_unique<pfaff_context>(*other.pfaff_);
    lncoshthetas_ = other.lncoshthetas_;
    coshthetas_ = other.coshthetas_;
    did_lncoshthetas_ = other.did_lncoshthetas_;
    did_coshthetas_ = other.did_coshthetas_;
    cosh_mode_ = other.cosh_mode_;
    init_cosh_funcs();
    return *this;
}

void rbm_context::init_cosh_funcs() {
    cosh_ = (cosh_mode_ == 1) ? math::cosh2 : math::cosh1;
    if (cosh_mode_ == 2)
        lncosh_ = math::lncosh;
    else
        lncosh_ = std::bind(&rbm_context::lncosh_default, this,
                            std::placeholders::_1);
}

pfaff_context& rbm_context::pfaff() {
    if (!pfaff_) pfaff_ = std::make_unique<pfaff_context>();
    return *pfaff_;
}

const pfaff_context& rbm_context::pfaff() const { return *pfaff_; }

Eigen::ArrayXXcd& rbm_context::coshthetas() {
    if (!did_coshthetas_) {
        if (did_lncoshthetas_) {
            coshthetas_ = lncoshthetas_.exp();
        } else {
            cosh_(thetas, coshthetas_);
        }
        did_coshthetas_ = true;
    }
    if (mpi::master) {
        // int c = 0;
        // for (int i = 0; i < coshthetas_.size(); i++) {
        //     if (std::isnan(std::real(coshthetas_(i))) ||
        //         std::isnan(std::imag(coshthetas_(i))) ||
        //         std::isinf(std::real(coshthetas_(i))) ||
        //         std::isinf(std::imag(coshthetas_(i))) ||
        //         std::abs(std::real(coshthetas_(i))) < 1e-5) {
        //         // std::abs(std::imag(coshthetas_(i))) < 1e-12) {
        //         std::cout << coshthetas_(i) << ", " << thetas(i) << "; ";
        //         c++;
        //     }
        // }
        // if (c > 0) std::cout << std::endl;
    }
    return coshthetas_;
}

Eigen::ArrayXXcd& rbm_context::lncoshthetas() {
    if (!did_lncoshthetas_) {
        lncosh_(thetas, lncoshthetas_);
        did_lncoshthetas_ = true;
    }
    return lncoshthetas_;
}

void rbm_context::updated_thetas() {
    did_lncoshthetas_ = false;
    did_coshthetas_ = false;
}

inline Eigen::ArrayXXcd rbm_context::lncosh_default(
    const Eigen::MatrixXcd& mat) {
    return coshthetas().log();
}
