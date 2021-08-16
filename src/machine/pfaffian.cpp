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

#include <iostream>
//
#include <machine/pfaffian.hpp>
#include <math.hpp>
#include <tools/eigen_fstream.hpp>
#include <tools/mpi.hpp>

using namespace machine;

pfaffian::pfaffian(const lattice::bravais& lattice, size_t n_sy)
    : lattice_{lattice},
      ns_{lattice.n_total},
      n_symm_{n_sy ? n_sy * n_sy * lattice.n_basis : ns_},
      fs_(4 * (lattice.n_total - 1), n_symm_),
      bs_(ns_, ns_),
      ss_(ns_, 1) {
    size_t n_uc = lattice_.n_uc;
    size_t n_tuc = lattice_.n_total_uc;
    size_t n_b = lattice_.n_basis;

    if (n_sy == 0) {
        n_sy = n_uc;
    }

    size_t uci, ucj, bi, bj, xi, yi, x, y;

    for (size_t i = 0; i < ns_; i++) {
        uci = lattice_.uc_idx(i);
        bi = lattice_.b_idx(i);
        xi = uci / n_uc;
        yi = uci % n_uc;
        ss_(i) = ((xi % n_sy) * n_sy + yi % n_sy) * n_b + bi;
        // ss_(i) = ((xi % n_sy) * n_sy + yi % n_sy);
        for (size_t j = 0; j < ns_; j++) {
            ucj = lattice_.uc_idx(j);
            bj = lattice_.b_idx(j);

            x = (ucj / n_uc - xi + n_uc) % n_uc;
            y = (ucj % n_uc - yi + n_uc) % n_uc;

            bs_(i, j) = n_b * ((x * n_uc + y) % n_tuc) + (bi ^ bj) - 1;
            // bs_(i, j) = ((x * n_uc + y) % n_tuc) - 1;
        }
    }
}

void pfaffian::init_weights(std::mt19937& rng, double std, bool normalize) {
    if (mpi::master) {
        std::normal_distribution<double> dist{0, std};
        std::complex<double> val;
        for (size_t j = 0; j < (size_t)fs_.cols(); j++) {
            for (size_t i = 0; i < (size_t)fs_.rows() / 4; i++) {
                val = std::complex<double>(dist(rng), dist(rng));
                for (size_t m = 0; m < 4; m++) {
                    fs_(i + m * (ns_ - 1), j) = val;
                }
            }
        }

        if (normalize) {
            Eigen::MatrixXcd mat = get_mat(Eigen::MatrixXcd::Ones(ns_, 1));
            int exp;
            math::pfaffian10(mat, exp);
            fs_ /= std::pow(10, (2.0 * exp) / ns_);
        }
    }
    MPI_Bcast(fs_.data(), fs_.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}

Eigen::MatrixXcd pfaffian::get_mat(const Eigen::MatrixXcd& state) const {
    Eigen::MatrixXcd mat(ns_, ns_);
    mat.setZero();
    for (size_t i = 0; i < ns_; i++) {
        for (size_t j = 0; j < i; j++) {
            mat(i, j) = a(i, j, state);
        }
    }
    mat.triangularView<Eigen::StrictlyUpper>() =
        mat.triangularView<Eigen::StrictlyLower>().transpose();
    mat.triangularView<Eigen::StrictlyUpper>() *= -1;
    return mat;
}

pfaff_context pfaffian::get_context(const Eigen::MatrixXcd& state) const {
    pfaff_context ret;

    Eigen::MatrixXcd mat = get_mat(state);

    ret.inv = mat.inverse();
    ret.inv -= ret.inv.transpose().eval();
    ret.inv /= 2;

    ret.pfaff = math::pfaffian10(mat, ret.exp);

    return ret;
}

void pfaffian::update_context(const Eigen::MatrixXcd& state,
                              const std::vector<size_t>& flips,
                              pfaff_context& context) const {
    if (flips.size() == 0) return;

    size_t m = flips.size();
    double sign = ((m * (m + 1) / 2) % 2 == 0 ? 1 : -1);
    Eigen::MatrixXcd B(ns_, 2 * m);
    B.setZero();
    Eigen::MatrixXcd Cinv(2 * m, 2 * m);
    Cinv.setZero();

    Cinv.block(m, 0, m, m).setIdentity();

    std::vector<bool> flipi(ns_);
    std::fill(flipi.begin(), flipi.end(), false);
    for (auto& f : flips) {
        flipi[f] = true;
    }

    for (size_t k = 0; k < m; k++) {
        B(flips[k], m + k) = 1;
        for (size_t i = 0; i < ns_; i++) {
            if (flips[k] != i) {
                B(i, k) = a(i, flips[k], state, flipi[i], true) -
                          a(i, flips[k], state);
            }
        }

        for (size_t l = 0; l < k; l++) {
            Cinv(k, l) = -a(flips[k], flips[l], state, true, true) +
                         a(flips[k], flips[l], state);
        }
    }

    Cinv.triangularView<Eigen::StrictlyUpper>() =
        Cinv.triangularView<Eigen::StrictlyLower>().transpose();
    Cinv.triangularView<Eigen::StrictlyUpper>() *= -1;

    Eigen::MatrixXcd invB = context.inv * B;

    Eigen::MatrixXcd tmp = (B.transpose() * invB) / 2;
    Cinv.noalias() += tmp;
    Cinv.noalias() -= tmp.transpose();

    Eigen::MatrixXcd C = Cinv.inverse() / 2;
    C -= C.transpose().eval();
    tmp = invB * C / 2;
    tmp *= invB.transpose();
    context.inv.noalias() += tmp;
    context.inv.noalias() -= tmp.transpose();

    std::complex<double> c2 = math::pfaffian(Cinv);

    c2 *= sign;  // * std::pow(10, p);
    context.pfaff *= c2;
    // context.exp += exp;
    context.update_factor = c2;
}

void pfaffian::derivative(const Eigen::MatrixXcd& state,
                          const pfaff_context& context,
                          Eigen::MatrixXcd& result, size_t& offset) const {
    Eigen::MatrixXcd d(fs_.rows(), fs_.cols());
    d.setZero();
    std::complex<double> x;
    for (size_t i = 0; i < ns_; i++) {
        for (size_t j = 0; j < i; j++) {
            x = context.inv(i, j);
            // if (state(i) != state(j)) x /= 2;
            d(idx(i, j, state), ss_(i)) = -x;
            d(idx(j, i, state), ss_(j)) = x;
        }
    }
    result.block(offset, 0, get_n_params(), 1) =
        Eigen::Map<Eigen::MatrixXcd>(d.data(), get_n_params(), 1);
    offset += get_n_params();
}

void pfaffian::update_weights(const Eigen::MatrixXcd& dw, size_t& offset) {
    fs_ -= Eigen::Map<const Eigen::MatrixXcd>(
        dw.block(offset, 0, get_n_params(), 1).data(), fs_.rows(), fs_.cols());
    offset += get_n_params();
}

void pfaffian::load(std::ifstream& input) {
    input >> fs_;
    // std::cout << fs_ << std::endl;
}
void pfaffian::save(std::ofstream& output) {
    output << fs_;
    // std::cout << fs_ << std::endl;
}

bool pfaffian::load_from_pfaffian_psi(const std::string& filename) {
    std::ifstream input{filename + ".rbm", std::ios::binary};
    if (input.good()) {
        // Read the n_updates_ from the inputstream.
        int n_updates;
        input.read((char*)&n_updates, sizeof(size_t));

        load(input);

        input.close();

        // Give a status update.
        std::cout << "Loaded Pfaffian from '" << filename << ".rbm'!"
                  << std::endl;
        return true;
    } else {
        return false;
    }
}
