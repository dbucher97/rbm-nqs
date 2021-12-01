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

#include <Eigen/Dense>
//
#include <lattice/honeycomb.hpp>
#include <lattice/honeycomb_hex.hpp>
#include <model/kitaev.hpp>
#include <operators/local_op.hpp>
#include <operators/local_op_chain.hpp>

using namespace model;

kitaev::kitaev(size_t size, const std::array<double, 3>& J, int size_b,
               const std::vector<double>& symm, bool hex_base, double h)
    : J{J}, h{h}, is_hex_{hex_base} {
    if (hex_base) {
        lattice_ = std::make_unique<lattice::honeycomb_hex>(size, symm);
    } else {
        lattice_ = std::make_unique<lattice::honeycomb>(size, size_b, symm);
    }
    std::vector<SparseXcd> bond_ops = {
        J[0] * kron({sx(), sx()}),
        J[1] * kron({sy(), sy()}),
        J[2] * kron({sz(), sz()}),
    };
    hamiltonian_ =
        std::make_unique<operators::bond_op>(lattice_->get_bonds(), bond_ops);
    if (std::abs(h) > 1e-10) {
        SparseXcd op = -h * sx() + h * sy() + h * sz();
        for (size_t i = 0; i < lattice_->n_total; i++) {
            hamiltonian_->push_back(operators::local_op({i}, op));
        }
    }
}

void kitaev::add_helper_hamiltonian(double strength) {
    auto hex =
        dynamic_cast<lattice::honeycomb*>(lattice_.get())->get_hexagons();
    SparseXcd plaq_op = kron({sx(), sy(), sz(), sx(), sy(), sz()});
    SparseXcd idn(plaq_op.rows(), plaq_op.cols());
    idn.setIdentity();
    plaq_op -= idn;
    plaq_op *= -strength;
    helpers_ = hex.size();
    for (auto& h : hex) {
        hamiltonian_->push_back(operators::local_op(h, plaq_op));
    }
}

double kitaev::exact_energy() const {
    double ret = 0;
    double smret = -1e10;
    double na, nb;
    na = lattice_->n_uc_b;
    nb = lattice_->n_uc;
    // Iterate through periodic / antiperiodic boundary conditions and pick
    // smallest energy
    for (int ci1 = 0; ci1 < 2; ci1++) {
        for (int ci2 = 0; ci2 < 2; ci2++) {
            double c1 = 0.5 / nb * ci1;
            double c2 = 0.5 / na * ci2;
            for (size_t i = 0; i < nb; i++) {
                for (size_t j = 0; j < na; j++) {
                    double fa = i / nb;
                    double fb = j / na;
                    if (is_hex_) fa -= fb;
                    double x = std::abs(f(fa + c1, fb + c2));
                    ret += x;
                }
            }
            if (ret > smret) {
                smret = ret;
            }
            ret = 0;
        }
    }
    return -0.5 * smret / lattice_->n_total;
}

std::complex<double> kitaev::f(double p1, double p2) const {
    return 2. *
           (J[0] * std::exp(std::complex<double>(0, 2 * M_PI * p1)) +
            J[1] * std::exp(std::complex<double>(0, 2 * M_PI * p2)) + J[2]);
}
