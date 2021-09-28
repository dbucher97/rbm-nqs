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

#include <Eigen/IterativeLinearSolvers>
#include <iostream>
#include <unsupported/Eigen/IterativeSolvers>
//
#include <machine/pfaffian.hpp>
#include <math.hpp>
#include <tools/eigen_fstream.hpp>
#include <tools/mpi.hpp>
#include <tools/time_keeper.hpp>

using namespace machine;

pfaffian::pfaffian(const lattice::bravais& lattice, size_t n_sy)
    : lattice_{lattice},
      ns_{lattice.n_total},
      n_symm_{n_sy ? n_sy * n_sy * lattice.n_basis : ns_},
      fs_(lattice.n_total * lattice.n_total, 4) {
    // size_t n_uc = lattice_.n_uc;
    // size_t n_tuc = lattice_.n_total_uc;
    // size_t n_b = lattice_.n_basis;

    // if (n_sy == 0) {
    //     n_sy = n_uc;
    // }

    // size_t uci, ucj, bi, bj, xi, yi, x, y;

    // for (size_t i = 0; i < ns_; i++) {
    //     uci = lattice_.uc_idx(i);
    //     bi = lattice_.b_idx(i);
    //     xi = uci / n_uc;
    //     yi = uci % n_uc;
    //     ss_(i) = ((xi % n_sy) * n_sy + yi % n_sy) * n_b + bi;
    //     // ss_(i) = ((xi % n_sy) * n_sy + yi % n_sy);
    //     for (size_t j = 0; j < ns_; j++) {
    //         ucj = lattice_.uc_idx(j);
    //         bj = lattice_.b_idx(j);

    //         x = (ucj / n_uc - xi + n_uc) % n_uc;
    //         y = (ucj % n_uc - yi + n_uc) % n_uc;

    //         bs_(i, j) = n_b * ((x * n_uc + y) % n_tuc) + (bi ^ bj) - 1;
    //         // bs_(i, j) = ((x * n_uc + y) % n_tuc) - 1;
    //     }
    // }
}

void pfaffian::init_weights(std::mt19937& rng, double std, bool normalize) {
    if (mpi::master) {
        std::normal_distribution<double> dist{0, std};
        std::complex<double> val;
        for (size_t j = 0; j < (size_t)fs_.rows(); j++) {
            for (size_t m = 0; m < 4; m++) {
                val = std::complex<double>(dist(rng), dist(rng));
                fs_(j, m) = val;
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

void pfaffian::init_weights_hf(
    const std::vector<Eigen::SparseMatrix<std::complex<double>>>& mats,
    const std::vector<std::vector<size_t>>& acts_on) {
    if (mpi::master) {
        auto bonds = lattice_.get_bonds();
        Eigen::MatrixXcd phi(ns_, 2 * ns_);
        Eigen::VectorXd eps(ns_);
        std::vector<std::vector<size_t>> nns(ns_);
        std::vector<std::vector<Eigen::SparseMatrix<std::complex<double>>>> h(
            ns_);
        phi.setRandom();

        phi = phi.rowwise().normalized();

        auto cidx = [&](size_t iidx, size_t sidx) { return sidx * ns_ + iidx; };
        // auto phi2 = [&](size_t m, size_t i, size_t sigma) {
        //     phi(m, cidx(i, sigma));
        // };

        for (size_t i = 0; i < ns_; i++) {
            nns[i] = lattice_.nns(i);
            h[i] = std::vector<Eigen::SparseMatrix<std::complex<double>>>(
                lattice_.n_coordination);
        }

        for (size_t i = 0; i < acts_on.size(); i++) {
            size_t a = acts_on[i][0], b = acts_on[i][1];
            int n = std::find(nns[a].begin(), nns[a].end(), b) - nns[a].begin();
            h[a][n] = mats[i];
            n = std::find(nns[b].begin(), nns[b].end(), a) - nns[b].begin();
            h[b][n] = mats[i].transpose();
        }
        Eigen::MatrixXcd F = Eigen::MatrixXcd(2 * ns_, 2 * ns_);

        auto gen_mat = [&]() {
            F.setZero();
            for (size_t i = 0; i < ns_; i++) {
                for (size_t g = 0; g < lattice_.n_coordination; g++) {
                    for (int k = 0; k < h[i][g].outerSize(); k++) {
                        for (auto it = Eigen::SparseMatrix<
                                 std::complex<double>>::InnerIterator(h[i][g],
                                                                      k);
                             it; ++it) {
                            int r = it.row(), c = it.col();
                            size_t ra = (r >> 1) & 1;
                            size_t rb = r & 1;
                            size_t ca = (c >> 1) & 1;
                            size_t cb = c & 1;

                            F(cidx(i, rb), cidx(i, cb)) +=
                                it.value() *
                                (phi.col(cidx(nns[i][g], ra))
                                     .array()
                                     .conjugate() *
                                 phi.col(cidx(nns[i][g], ca)).array())
                                    .sum();
                            std::complex<double> x =
                                it.value() * (phi.col(cidx(nns[i][g], ra))
                                                  .array()
                                                  .conjugate() *
                                              phi.col(cidx(i, cb)).array())
                                                 .sum();
                            F(cidx(i, rb), cidx(nns[i][g], ca)) -= std::conj(x);
                            F(cidx(nns[i][g], ca), cidx(i, rb)) -= x;
                        }
                    }
                }
            }
        };

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver;

        for (size_t i = 0; i < 10000; i++) {
            gen_mat();
            solver.compute(F);
            phi = solver.eigenvectors()
                      .block(0, 0, phi.cols(), phi.rows())
                      .transpose();
        }

        Eigen::MatrixXcd Fb(2 * ns_, 2 * ns_);

        for (size_t i = 0; i < 2 * ns_; i++)
            for (size_t j = 0; j < 2 * ns_; j++) {
                Fb(i, j) = 0;
                for (size_t n = 0; n < ns_ / 2; n++) {
                    Fb(i, j) += phi(2 * n, i) * phi(2 * n + 1, j) -
                                phi(2 * n, j) * phi(2 * n + 1, i);
                }
            }

        for (size_t i = 0; i < ns_; i++) {
            for (size_t j = 0; j < ns_; j++) {
                for (size_t si = 0; si < 2; si++) {
                    for (size_t sj = 0; sj < 2; sj++) {
                        fs_(j * ns_ + i, 2 * si + sj) =
                            Fb(cidx(i, si), cidx(j, sj));
                        -Fb(cidx(j, sj), cidx(i, si));
                    }
                }
            }
        }
        std::cout << fs_ << std::endl;

        // Eigen::MatrixXcd mat = Fb.block(0, 0, ns_, ns_);
        // mat -= mat.transpose().eval();

        // std::cout << mat << std::endl;
        // std::cout << get_mat(-Eigen::MatrixXcd::Ones(8, 1)) << std::endl;
    }
    bcast(0);
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
    ret.inv.noalias() -= ret.inv.transpose().eval();
    ret.inv *= 0.5;

    ret.pfaff = math::pfaffian10(mat, ret.exp);

    return ret;
}

void pfaffian::update_context(const Eigen::MatrixXcd& state,
                              const std::vector<size_t>& flips,
                              pfaff_context& context) const {
    time_keeper::start("Pfaff Context");
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

    // Eigen::BiCGSTAB<Eigen::MatrixXcd> solver(mat);
    // solver.setMaxIterations(5);
    // Eigen::MatrixXcd invB3 = solver.solveWithGuess(B, invB);

    Eigen::MatrixXcd tmp = (B.transpose() * invB) * 0.5;
    Cinv.noalias() += tmp;
    Cinv.noalias() -= tmp.transpose();

    Eigen::MatrixXcd C = 0.5 * Cinv.inverse();
    C -= C.transpose().eval();
    tmp = invB * C;
    tmp = 0.5 * tmp * invB.transpose();
    context.inv.noalias() += tmp;
    context.inv.noalias() -= tmp.transpose();

    int exp;
    std::complex<double> c2 = math::pfaffian10(Cinv, exp);

    c2 *= sign;  // * std::pow(10, p);

    context.pfaff *= c2;
    context.exp += exp;
    context.update_factor = c2 * std::pow(10, exp);

    int s = std::log10(std::abs(context.pfaff));

    if (std::abs(s) > 1) {
        context.pfaff *= std::pow(10, -s);
        context.exp += s;
    }

    time_keeper::end("Pfaff Context");
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
            d(idx(i, j), spidx(i, j, state)) = -x;
            d(idx(j, i), spidx(j, i, state)) = x;
        }
    }
    result.block(offset, 0, get_n_params(), 1) =
        Eigen::Map<Eigen::MatrixXcd>(d.data(), get_n_params(), 1);
    offset += get_n_params();
}

void pfaffian::update_weights(const Eigen::MatrixXcd& dw, size_t& offset) {
    fs_ -= Eigen::Map<const Eigen::MatrixXcd>(
        dw.block(offset, 0, get_n_params(), 1).data(), fs_.rows(), fs_.cols());
    // auto x = Eigen::Map<const Eigen::MatrixXcd>(
    //     dw.block(offset, 0, get_n_params(), 1).data(), fs_.rows(),
    //     fs_.cols());
    if (mpi::master) {
        // for (size_t i = 0; i < 4; i++) {
        //     std::cout << "\n"
        //               << Eigen::Map<const
        //               Eigen::MatrixXcd>(x.col(i).data(),
        //                                                     ns_, ns_)
        //               << "\n"
        //               << std::endl;
        // }
    }
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
    bool rc = false;
    if (mpi::master) {
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
            rc = true;
        } else {
            rc = false;
        }
    }
    MPI_Bcast(&rc, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    if (rc) {
        bcast(0);
    }
    return rc;
}

void pfaffian::bcast(int rank) {
    MPI_Bcast(fs_.data(), fs_.size(), MPI_DOUBLE_COMPLEX, rank, MPI_COMM_WORLD);
}

bool pfaffian::spidx(size_t i, const Eigen::MatrixXcd& state, bool flip) const {
    return (std::real(state(i)) < 0) ^ flip;
}
int pfaffian::spidx(size_t i, size_t j, const Eigen::MatrixXcd& state,
                    bool flipi, bool flipj) const {
    return (spidx(i, state, flipi) << 1) + spidx(j, state, flipj);
}
size_t pfaffian::idx(size_t i, size_t j) const { return i * ns_ + j; }

std::complex<double> pfaffian::a(size_t i, size_t j,
                                 const Eigen::MatrixXcd& state, bool flipi,
                                 bool flipj) const {
    if (i == j) {
        return 0;
    }
    return fs_(idx(i, j), spidx(i, j, state, flipi, flipj)) -
           fs_(idx(j, i), spidx(j, i, state, flipj, flipi));
}
