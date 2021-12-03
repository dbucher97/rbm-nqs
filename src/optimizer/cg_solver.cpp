/*
 * Copyright (C) 2021  David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * */

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <iostream>
//
#include <optimizer/cg_solver.hpp>
#include <tools/ini.hpp>
#include <tools/mpi.hpp>
#include <tools/time_keeper.hpp>

using namespace optimizer;

cg_method get_method(const std::string& s, size_t n, size_t m, size_t t) {
    if (s == "cg" || s == "minres") {
        // MINRES NOT YET IMPLEMENTED
        // Used here to go directly to distributed CG
        return CG;
    } else if (s == "cg-single") {
        return CG_SINGLE;
    } else {
        size_t flops_single = 2 * n * n * (m + t) - n * (n + t);
        size_t flops_dist = 4 * n * m * t - t * (m + n);
        mpi::cout << "using CG method: ";
        mpi::cout << (flops_dist < flops_single ? "CG" : "CG_SINGLE")
                  << mpi::endl;
        return (flops_dist < flops_single ? CG : CG_SINGLE);
    }
}

cg_solver::cg_solver(size_t n, size_t m, size_t n_neural, size_t max_iterations,
                     double rtol, const std::string& method)
    : Base{n, n_neural},
      max_iterations_{std::min(2 * n, max_iterations)},
      rtol_{rtol},
      method_{get_method(method, n, m, max_iterations_)} {
    if (max_iterations_ == 0) {
        max_iterations_ = 2 * n;
    }
    if (rtol_ == 0.) {
        rtol_ = 1e-16;
    }

    if (mpi::master) {
        r_ = Eigen::VectorXcd(n);
    }
    Ap_ = Eigen::VectorXcd(n);
    p_ = Eigen::VectorXcd(n);
    if (method_ == MINRES) {
        Ap2_ = Eigen::VectorXcd(n);
        p2_ = Eigen::VectorXcd(n);
    }
}

void cg_solver::cg_single(const std::function<void(const Eigen::VectorXcd&,
                                                   Eigen::VectorXcd&)>& Aprod,
                          const Eigen::VectorXcd& b, Eigen::VectorXcd& x) {
    if (mpi::master) {
        int omp_prev = omp_get_max_threads();
        omp_set_num_threads(ini::n_threads);
        double rsold;
        bool abort;
        std::complex<double> alpha;

        r_ = b;
        if (x.size() == 0) {
            x = Eigen::VectorXcd(b.size());
        } else {
            Aprod(x, Ap_);
            r_ -= Ap_;
        }

        p_ = r_;
        rsold = r_.squaredNorm();

        for (itn_ = 0; itn_ < max_iterations_; itn_++) {
            if (itn_ > 0) time_keeper::end("CG step");
            Aprod(p_, Ap_);
            time_keeper::start("CG step");
            alpha = p_.dot(Ap_);

            alpha = rsold / alpha;
            x += alpha * p_;
            r_ -= alpha * Ap_;
            rs_ = r_.squaredNorm();
            abort = rs_ < rtol_;
            // if (mpi_rank == 0)
            //     std::cout << i << ", " << std::sqrt(rsnew) << std::endl;
            if (abort) break;
            p_ = r_ + (rs_ / rsold) * p_;
            rsold = rs_;
        }
        time_keeper::end("CG step");
        omp_set_num_threads(omp_prev);
    }

    time_keeper::start("Opt MPI");
    MPI_Bcast(x.data(), x.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    time_keeper::end("Opt MPI");
}

void cg_solver::cg1(const std::function<void(const Eigen::VectorXcd&,
                                             Eigen::VectorXcd&)>& Aprod,
                    const Eigen::VectorXcd& b, Eigen::VectorXcd& x) {
    double rsold;
    bool abort;
    std::complex<double> alpha;

    if (mpi::master) r_ = b;
    if (x.size() == 0) {
        x = Eigen::VectorXcd(b.size());
    } else {
        Aprod(x, Ap_);
        if (mpi::master) {
            MPI_Reduce(MPI_IN_PLACE, Ap_.data(), Ap_.size(), MPI_DOUBLE_COMPLEX,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(Ap_.data(), Ap_.data(), Ap_.size(), MPI_DOUBLE_COMPLEX,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        }
        if (mpi::master) {
            r_ -= Ap_;
        }
    }

    if (mpi::master) {
        p_ = r_;
        rsold = r_.squaredNorm();
    }

    // int omp_prev = omp_get_max_threads();
    for (itn_ = 0; itn_ < max_iterations_; itn_++) {
        time_keeper::start("Opt MPI");
        MPI_Bcast(p_.data(), p_.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
        time_keeper::end("Opt MPI");
        if (itn_ > 0) time_keeper::end("CG step");
        Aprod(p_, Ap_);
        time_keeper::start("CG step");
        time_keeper::start("Opt MPI");
        if (mpi::master) {
            MPI_Reduce(MPI_IN_PLACE, Ap_.data(), Ap_.size(), MPI_DOUBLE_COMPLEX,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(Ap_.data(), Ap_.data(), Ap_.size(), MPI_DOUBLE_COMPLEX,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        }
        time_keeper::end("Opt MPI");
        if (mpi::master) {
            // omp_set_num_threads(ini::n_threads);
            alpha = p_.dot(Ap_);

            alpha = rsold / alpha;
            x += alpha * p_;
            r_ -= alpha * Ap_;
            rs_ = r_.squaredNorm();
            abort = rs_ < rtol_;
        }
        time_keeper::start("Opt MPI");
        MPI_Bcast(&abort, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        time_keeper::end("Opt MPI");
        // if (mpi_rank == 0)
        //     std::cout << i << ", " << std::sqrt(rsnew) << std::endl;
        if (abort) break;
        if (mpi::master) {
            p_ = r_ + (rs_ / rsold) * p_;
            rsold = rs_;
            // omp_set_num_threads(omp_prev);
        }
    }
    time_keeper::end("CG step");
    MPI_Bcast(x.data(), x.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    // if (mpi::master) {
    //     omp_set_num_threads(omp_prev);
    // }
}

void cg_solver::minres(const std::function<void(const Eigen::VectorXcd&,
                                                Eigen::VectorXcd&)>& Aprod,
                       const Eigen::VectorXcd& b, Eigen::VectorXcd& x) {
    double rsold;
    bool abort;
    std::complex<double> alpha;

    if (mpi::master) r_ = b;
    if (x.size() == 0) {
        x = Eigen::VectorXcd(b.size());
    } else {
        Aprod(x, Ap_);
        if (mpi::master) {
            MPI_Reduce(MPI_IN_PLACE, Ap_.data(), Ap_.size(), MPI_DOUBLE_COMPLEX,
                       MPI_SUM, 0, MPI_COMM_WORLD);
            r_ -= Ap_;
        } else {
            MPI_Reduce(Ap_.data(), Ap_.data(), Ap_.size(), MPI_DOUBLE_COMPLEX,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }

    if (mpi::master) {
        p_ = r_;
        rsold = r_.squaredNorm();
    }

    int omp_prev = omp_get_max_threads();
    for (itn_ = 0; itn_ < max_iterations_; itn_++) {
        MPI_Bcast(p_.data(), p_.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
        Aprod(p_, Ap_);
        if (mpi::master) {
            MPI_Reduce(MPI_IN_PLACE, Ap_.data(), Ap_.size(), MPI_DOUBLE_COMPLEX,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(Ap_.data(), Ap_.data(), Ap_.size(), MPI_DOUBLE_COMPLEX,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        }
        if (mpi::master) {
            omp_set_num_threads(ini::n_threads);
            alpha = p_.dot(Ap_);

            alpha = rsold / alpha;
            x += alpha * p_;
            r_ -= alpha * Ap_;
            rs_ = r_.squaredNorm();
            abort = rs_ < rtol_;
        }
        MPI_Bcast(&abort, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        // if (mpi_rank == 0)
        //     std::cout << i << ", " << std::sqrt(rsnew) << std::endl;
        if (abort) break;
        if (mpi::master) {
            p_ = r_ + (rs_ / rsold) * p_;
            rsold = rs_;
            omp_set_num_threads(omp_prev);
        }
    }
    MPI_Bcast(x.data(), x.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    if (mpi::master) {
        omp_set_num_threads(omp_prev);
    }
}

void cg_solver::solve(const Eigen::MatrixXcd& mat, const Eigen::VectorXcd& d,
                      const double norm, const Eigen::VectorXcd& b,
                      Eigen::VectorXcd& x, const double r1, const double r2,
                      const double rd, const Eigen::VectorXd& diag) {
    // if (mpi::master)
    //     std::cout << "\n" << r1 << ", " << r2 << ", " << rd << std::endl;
    int nnn = n_ - n_neural_;
    double max_diag = diag.maxCoeff();
    // max_diag = 1;

    Eigen::MatrixXcd S;

    std::function<void(const Eigen::VectorXcd&, Eigen::VectorXcd&)> Aprod;

    if (method_ == CG_SINGLE) {
        time_keeper::start("Mat gen");
        S = mat.conjugate() * mat.transpose() / norm;
        if (mpi::rank == mpi::n_proc - 1) {
            S.diagonal().topRows(n_neural_) += r1 * diag.topRows(n_neural_);
            S.diagonal().bottomRows(nnn) += (r1 + rd) * diag.bottomRows(nnn);
            S.diagonal().array() += max_diag * r2;
            S -= d.conjugate() * d.transpose();
        }
        time_keeper::end("Mat gen");

        time_keeper::start("Opt MPI");
        if (mpi::rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, S.data(), S.size(), MPI_DOUBLE_COMPLEX,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(S.data(), S.data(), S.size(), MPI_DOUBLE_COMPLEX,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        }
        time_keeper::end("Opt MPI");

        Aprod = [&](const Eigen::VectorXcd& b, Eigen::VectorXcd& x) {
            time_keeper::start("Matmul");
            x = S * b;
            time_keeper::end("Matmul");
        };
    } else {
        Aprod = [&](const Eigen::VectorXcd& b, Eigen::VectorXcd& x) {
            // Use last process, since the front processes are likely to be
            // more busy, since they are assinged more samples.
            time_keeper::start("Matmul");
            if (mpi::rank == mpi::n_proc - 1) {
                x = diag.array();
                x.topRows(n_neural_) *= r1;
                x.bottomRows(nnn) *= r1 + rd;

                x.array() += max_diag * r2;
                x.array() *= b.array();
                x += mat.conjugate() * (mat.transpose() * b) / norm;
                x -= d.conjugate() * (d.transpose() * b);
            } else {
                x = mat.conjugate() * (mat.transpose() * b) / norm;
            }
            time_keeper::end("Matmul");
        };
    }

    switch (method_) {
        case CG:
        case MINRES:
            cg1(Aprod, b, x);
            break;
        case CG_SINGLE:
            cg_single(Aprod, b, x);
            break;
    }
    // mpi::cout << "rnorm: " << rs_ << " \tn_iter: " << itn_
    //           << " \tmaxdiag: " << max_diag << " \tr2:" << r2 << mpi::endl;
    // if (mpi::master) std::cout << std::endl << rs_ << ", " << itn_ <<
    // std::endl;
}

// void cg_solver::split_blocks(size_t total, size_t bs, int* lens) {
//     size_t n_blocks = total / bs;
//     size_t n_blocks_assigned = 0;
//     size_t bs_last = total - bs * n_blocks;
//     size_t total_len = 0;

//     for (int i = 0; i < mpi::n_proc; i++) {
//         lens[i] = bs * (n_blocks / mpi::n_proc);
//         n_blocks_assigned += n_blocks / mpi::n_proc;
//         total_len += lens[i];
//     }

//     if (n_blocks_assigned != n_blocks || n_blocks == 0) {
//         int n_new = n_blocks + (bs_last > 0);
//         if (n_new > mpi::n_proc) {
//             n_new = mpi::n_proc;
//         }
//         size_t x = (total - total_len) / n_new;
//         for (int i = 0; i < n_new; i++) {
//             lens[i] += x;
//             total_len += x;
//         }

//         size_t i = 0;
//         while (total_len < total) {
//             lens[i]++;
//             total_len++;
//             i++;
//         }
//     }
// }

// void cg_solver::cg(const std::function<void(const Eigen::VectorXcd&,
//                                             Eigen::VectorXcd&)>& Aprod,
//                    const Eigen::VectorXcd& b, Eigen::VectorXcd& x) {
//     int* lens = new int[mpi::n_proc];
//     int* starts = new int[mpi::n_proc];
//     split_blocks(b.size(), bs_, lens);
//     size_t start = 0;
//     size_t len = lens[mpi::rank];
//     int z_proc = 0;
//     // if (mpi_rank == 0) std::cout << "process distribution" << std::endl;
//     for (int i = 0; i < mpi::n_proc; i++) {
//         starts[i] = start;
//         if (lens[i]) {
//             start += lens[i];
//             z_proc++;
//         }
//         // if (mpi_rank == 0) {
//         //     std::cout << i << ", " << lens[i] << std::endl;
//         // }
//     }
//     start = starts[mpi::rank];

//     int* nz_procs = new int[z_proc];
//     int* z_procs = new int[mpi::n_proc - z_proc + 1];
//     z_procs[0] = 0;

//     for (int i = 0; i < mpi::n_proc; i++) {
//         if (i < z_proc) {
//             nz_procs[i] = i;
//         } else {
//             z_procs[i - z_proc + 1] = i;
//         }
//     }

//     // if (mpi_rank == 0) {
//     //     std::cout << "NZ Comm" << std::endl;
//     //     for (size_t i = 0; i < z_proc; i++) std::cout << nz_procs[i] << ",
//     ";
//     //     std::cout << "\nZ Comm" << std::endl;
//     //     for (size_t i = 0; i < n_proc - z_proc + 1; i++)
//     //         std::cout << z_procs[i] << ", ";
//     //     std::cout << std::endl;
//     // }

//     MPI_Group nz_group, z_group, w_group;
//     MPI_Comm nz_comm, z_comm;
//     MPI_Comm_group(MPI_COMM_WORLD, &w_group);
//     MPI_Group_incl(w_group, z_proc, nz_procs, &nz_group);
//     MPI_Group_incl(w_group, mpi::n_proc - z_proc + 1, z_procs, &z_group);
//     MPI_Comm_create_group(MPI_COMM_WORLD, z_group, 0, &z_comm);
//     MPI_Comm_create_group(MPI_COMM_WORLD, nz_group, 1, &nz_comm);

//     Eigen::VectorXcd r, lx, tmp, p(len), Ap(len), gp(b.size());

//     if (len) r = b.segment(start, len);

//     if (x.size() == 0) {
//         x = Eigen::VectorXcd(b.size());
//         lx = Eigen::VectorXcd(len);
//         lx.setZero();
//     } else {
//         Aprod(x, tmp);
//         MPI_Reduce_scatter(tmp.data(), Ap.data(), lens, MPI_DOUBLE_COMPLEX,
//                            MPI_SUM, MPI_COMM_WORLD);
//         if (len) {
//             r -= Ap;
//             lx = x.segment(start, len);
//         }
//     }

//     p = r;
//     double rsold, rsnew, lr = 0;
//     bool abort;
//     std::complex<double> la, alpha;
//     if (len) {
//         lr = r.squaredNorm();
//         MPI_Allreduce(&lr, &rsold, 1, MPI_DOUBLE, MPI_SUM, nz_comm);
//     }

//     for (itn_ = 0; itn_ < max_iterations_; itn_++) {
//         MPI_Allgatherv(p.data(), p.size(), MPI_DOUBLE_COMPLEX, gp.data(),
//         lens,
//                        starts, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
//         Aprod(gp, tmp);
//         MPI_Reduce_scatter(tmp.data(), Ap.data(), lens, MPI_DOUBLE_COMPLEX,
//                            MPI_SUM, MPI_COMM_WORLD);
//         if (len) {
//             la = p.dot(Ap);
//             MPI_Allreduce(&la, &alpha, 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
//             nz_comm); alpha = rsold / alpha; lx += alpha * p; r -= alpha *
//             Ap; lr = r.squaredNorm(); MPI_Allreduce(&lr, &rsnew, 1,
//             MPI_DOUBLE, MPI_SUM, nz_comm); abort = std::sqrt(rsnew) < rtol_;
//         }
//         if (!len || mpi::rank == 0) {
//             MPI_Bcast(&abort, 1, MPI_CXX_BOOL, 0, z_comm);
//         }
//         // if (mpi_rank == 0)
//         //     std::cout << i << ", " << std::sqrt(rsnew) << std::endl;
//         if (abort) break;
//         if (len) {
//             p = r + (rsnew / rsold) * p;
//             rsold = rsnew;
//         }
//     }
//     if (len)
//         MPI_Allgatherv(lx.data(), lx.size(), MPI_DOUBLE_COMPLEX, x.data(),
//         lens,
//                        starts, MPI_DOUBLE_COMPLEX, nz_comm);
//     if (!len || mpi::rank == 0) {
//         MPI_Bcast(x.data(), x.size(), MPI_DOUBLE_COMPLEX, 0, z_comm);
//     }

//     MPI_Group_free(&w_group);
//     MPI_Group_free(&z_group);
//     MPI_Group_free(&nz_group);
//     if (!len || mpi::rank == 0) MPI_Comm_free(&z_comm);
//     if (len) MPI_Comm_free(&nz_comm);

//     delete[] lens;
//     delete[] starts;
//     delete[] z_procs;
//     delete[] nz_procs;
// }
