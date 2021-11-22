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

pfaffian::pfaffian(const lattice::bravais& lattice,
                   const std::vector<double>& symm, bool no_updating)
    : lattice_{lattice},
      ns_{lattice.n_total},
      n_symm_{lattice_.symmetry_size(symm)},
      symmetry_{lattice_.construct_symmetry(symm)},
      symm_basis_{lattice_.construct_symm_basis(symm)},
      bs_(ns_, ns_),
      no_updating_{no_updating} {
    bs_.setZero();
    int offset = 0;
    // This procedure is not very efficient, but does not need to be.
    // Fill Index Matrix with independent indices for each symmetry
    for (auto& bi : symm_basis_) {
        for (auto& s : symmetry_) {
            auto& idcs = s.indices();
            int a, b;
            int f = 1;
            // Dirty way of checking if symmetry is mirrord (only for hex)
            if (idcs(0) > idcs(1)) {
                f = -1;
            }
            for (size_t j = 0; j < ns_; j++) {
                a = idcs(bi);
                b = idcs(j);
                if (b == a)
                    bs_(b, a) = 0;
                else
                    bs_(b, a) = f * (j + offset);
            }
        }
        offset += ns_;
    }
    // Check symmetry constraints due to transposition
    for (size_t i = 0; i < ns_; i++) {
        for (size_t j = 0; j < i; j++) {
            if (std::abs(bs_(i, j)) <= std::abs(bs_(j, i))) {
                bs_(j, i) = -bs_(i, j);
            } else {
                bs_(i, j) = -bs_(j, i);
            }
        }
    }
    // Condense indices into dense format.
    for (int i = 1; i < bs_.maxCoeff(); i++) {
        while ((bs_.array() == i).count() == 0) {
            bs_ = (bs_.array() > i).select(bs_.array() - 1, bs_.array());
            bs_ = (bs_.array() < -i).select(bs_.array() + 1, bs_.array());
        }
    }

    // Check symmetry constraints due to transposition
    // bs_.triangularView<Eigen::StrictlyUpper>() *= -1;

    fs_.resize(bs_.maxCoeff(), 4);

    // if (symmetry_.size() == ns_) {
    //     for (auto& s : symmetry_) {
    //         int r = s.indices()(0);
    //         if (r > s.indices()(1)) {
    //             bs_.col(r).bottomRows(ns_ - r) *= -1;
    //             bs_.row(r).transpose().bottomRows(ns_ - r) *= -1;
    //         }
    //     }
    // }

    // bs_ -= bs_.transpose().eval();
    // } else {
    //     for (size_t i = 0; i < ns_; i++) {
    //         for (size_t j = 0; j < i; j++) {
    //             bs_(i, j) = idx(i, j) + 1;
    //             bs_(j, i) = -idx(i, j) - 1;
    //         }
    //     }
    // }
    //
    //

    // mpi::cout << bs_ << mpi::endl;
}

bool pfaffian::spidx(size_t i, const spin_state& state, bool flip) const {
    return state[i] ^ flip;
}
int pfaffian::spidx(size_t i, size_t j, const spin_state& state, bool flipi,
                    bool flipj) const {
    return (spidx(i, state, flipi) << 1) + spidx(j, state, flipj);
}
size_t pfaffian::idx(size_t i, size_t j) const {
    if (i < j) {
        return idx(j, i);
    } else {
        return ns_ * (ns_ - 1) / 2 - (ns_ - j) * (ns_ - j - 1) / 2 + i - j - 1;
    }
}

std::complex<double> pfaffian::a(size_t i, size_t j, const spin_state& state,
                                 bool flipi, bool flipj) const {
    int bs = bs_(i, j);
    double sgn = bs < 0 ? -1. : 1.;
    int sp = (j < i) ? spidx(i, j, state, flipi, flipj)
                     : spidx(j, i, state, flipj, flipi);
    if (i == j)
        return 0;
    else
        return sgn * fs_(std::abs(bs) - 1, sp);
}

void pfaffian::init_weights(std::mt19937& rng, double std, bool normalize) {
    if (mpi::master) {
        std::normal_distribution<double> dist{0, std};
        fs_.setZero();
        for (int i = 0; i < fs_.size(); i++) {
            fs_(i) = std::complex<double>(dist(rng), dist(rng));
        }

        if (normalize) {
            Eigen::MatrixXcd mat = get_mat(spin_state(ns_));
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
    /*if (mpi::master) {
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
    bcast(0); */
}

Eigen::MatrixXcd pfaffian::get_mat(const spin_state& state) const {
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

pfaff_context pfaffian::get_context(const spin_state& state) const {
    pfaff_context ret;

    Eigen::MatrixXcd mat = get_mat(state);

    ret.inv = mat.completeOrthogonalDecomposition().pseudoInverse();
    // ret.inv = mat.inverse()
    ret.inv.noalias() -= ret.inv.transpose().eval();
    ret.inv *= 0.5;

    ret.pfaff = math::pfaffian10(mat, ret.exp);

    return ret;
}

void pfaffian::update_context(const spin_state& state,
                              const std::vector<size_t>& flips,
                              pfaff_context& context) const {
    if (flips.size() == 0) return;
    time_keeper::start("Pfaff Context");
    if (no_updating_) {
        spin_state state2 = state;
        state2.flip(flips);
        ;
        std::complex<double> pfaff = context.pfaff;
        int exp = context.exp;
        context = get_context(state2);
        context.update_factor =
            context.pfaff / pfaff * std::pow(10., context.exp - exp);
    } else {
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

        // auto mat = get_mat(state);
        // Eigen::BiCGSTAB<Eigen::MatrixXcd> solver(mat);
        // solver.setMaxIterations(5);
        // invB = solver.solveWithGuess(B, invB);

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

        // std::cout << context.pfaff << "e" << context.exp;
        // Eigen::MatrixXcd state2 = state;
        // for (auto& f : flips) state2(f) *= -1;
        // Eigen::MatrixXcd mat = get_mat(state2);
        // Eigen::JacobiSVD<Eigen::MatrixXcd> svd(mat);
        // double cond = svd.singularValues()(0) /
        //               svd.singularValues()(svd.singularValues().size() - 1);
        // std::cout << "\t cond " << cond;

        // Eigen::MatrixXcd mati = mat.inverse();
        // std::cout << "\t vanilla norm"
        //           << (mati - context.inv).norm() / mati.norm() << std::endl;

        int s = std::log10(std::abs(context.pfaff));

        if (std::abs(s) > 1) {
            context.pfaff *= std::pow(10, -s);
            context.exp += s;
        }
    }

    time_keeper::end("Pfaff Context");
}

void pfaffian::derivative(const spin_state& state, const pfaff_context& context,
                          Eigen::MatrixXcd& result, size_t& offset) const {
    Eigen::MatrixXcd d(fs_.rows(), fs_.cols());
    d.setZero();
    std::complex<double> x;
    for (size_t i = 0; i < ns_; i++) {
        for (size_t j = 0; j < i; j++) {
            x = context.inv(i, j);
            int bs = bs_(i, j);
            d(std::abs(bs) - 1, spidx(i, j, state)) += bs > 0 ? -x : x;
            // d(idx(j, i), spidx(j, i, state)) = x;
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
    // if (mpi::master) {
    //     for (size_t i = 0; i < 4; i++) {
    //         std::cout << "\n"
    //                   << Eigen::Map<const
    //                   Eigen::MatrixXcd>(x.col(i).data(),
    //                                                         ns_, ns_)
    //                          .transpose()
    //                   << "\n"
    //                   << std::endl;
    //     }
    // }
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
