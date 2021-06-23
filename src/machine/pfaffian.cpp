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

#include <machine/pfaffian.hpp>
#include <math.hpp>

using namespace machine;

pfaffian::pfaffian(const lattice::bravais& lattice, size_t n_sy)
    : lattice_{lattice},
      ns_{lattice.n_total},
      n_symm_{n_sy * n_sy * lattice.n_basis},
      fs_(4 * (lattice.n_total - 1), n_symm_),
      bs_(ns_, ns_),
      ss_(ns_, 1) {
    size_t n_uc = lattice_.n_uc;
    size_t n_tuc = lattice_.n_total_uc;
    size_t n_b = lattice_.n_basis;
    size_t n_b2 = n_b * n_b;

    int uci, ucj, bi, bj, xi, yi, x, y;

    for (size_t i = 0; i < ns_; i++) {
        uci = lattice_.uc_idx(i);
        bi = lattice_.b_idx(i);
        xi = uci / n_uc;
        yi = uci % n_uc;
        ss_(i) = ((xi + n_sy) % n_sy * n_sy) * n_b + bi;
        for (size_t j = 0; j < ns_; j++) {
            ucj = lattice_.uc_idx(j);
            bj = lattice_.b_idx(j);

            x = xi - ucj / n_uc;
            y = yi - ucj % n_uc + n_tuc;
            bs_(i, j) = n_b2 * ((x * n_uc + y) % n_tuc) + n_b * bi + bj;
        }
    }
}

pfaff_context pfaffian::get_context(const Eigen::MatrixXcd& state) const {
    pfaff_context ret;

    Eigen::MatrixXcd mat(ns_, ns_);
    mat.setZero();
    for (size_t i = 0; i < ns_; i++) {
        for (size_t j = 0; j < i; j++) {
            mat(i, j) = a(i, j, state);
        }
    }
    mat -= mat.transpose().eval();
    ret.inv = mat.inverse();
    ret.inv -= ret.inv.transpose().eval();
    ret.inv /= 2;

    ret.pfaff = math::pfaffian10(mat, ret.exp);

    return ret;
}

std::complex<double> pfaffian::update_context(const Eigen::MatrixXcd& state,
                                              const std::vector<size_t>& flips,
                                              pfaff_context& context) const {
    if (flips.size() == 0) return 1;

    size_t m = flips.size();
    double sign = ((m * (m + 1) / 2) % 2 == 0 ? 1 : -1);
    Eigen::MatrixXcd B(ns_, 2 * m);
    B.setZero();
    Eigen::MatrixXcd Cinv(2 * m, 2 * m);
    Cinv.setZero();

    Cinv.block(m, 0, m, m).setIdentity();

    bool flipi;

    for (size_t i = 0; i < ns_; i++) {
        flipi = std::find(flips.begin(), flips.end(), i) != flips.end();
        for (size_t k = 0; k < m; k++) {
            B(i, k) =
                a(i, flips[k], state, flipi, true) - a(i, flips[k], state);
            B(i, m + k) = (i == flips[k]) ? 1 : 0;
            for (size_t l = 0; l < k; l++) {
                Cinv(k, l) = -a(flips[k], flips[l], state, true, true) +
                             a(flips[k], flips[l], state);
            }
        }
    }
    Cinv -= Cinv.transpose().eval();

    Eigen::MatrixXcd invB = context.inv * B;

    Eigen::MatrixXcd tmp = (B.transpose() * invB);
    Cinv.noalias() += 0.5 * tmp - 0.5 * tmp.transpose();

    Eigen::MatrixXcd C = Cinv.inverse();
    C -= C.transpose().eval();
    C /= 2;
    tmp = invB * C * invB.transpose();
    context.inv.noalias() += 0.5 * tmp - 0.5 * tmp.transpose();

    std::complex<double> c2;
    skpfa(2 * m, Cinv.data(), &c2, "L", "P");

    c2 *= sign;
    context.pfaff *= c2;
    return c2;
}

Eigen::MatrixXcd pfaffian::derivative(const Eigen::MatrixXcd& state,
                                      const pfaff_context& context) const {
    Eigen::MatrixXcd d(fs_.rows(), fs_.cols());
    d.setZero();
    std::complex<double> x;
    for (size_t i = 0; i < ns_; i++) {
        for (size_t j = 0; j < i; j++) {
            x = context.inv(i, j);
            d(idx(i, j, state), ss_(i)) = -x;
            d(idx(j, i, state), ss_(j)) = x;
        }
    }
    return Eigen::Map<Eigen::MatrixXcd>(d.data(), fs_.size(), 1);
}

void pfaffian::update_weights(Eigen::MatrixXcd& dw) {}
