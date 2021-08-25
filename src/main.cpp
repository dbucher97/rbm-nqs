/*
 * Copyright (c) 2021 David Bucher <David.Bucher@physik.lmu.de> * * This program is free software: you can redistribute it and/or modify * it under the terms of the GNU Affero General Public License as * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARjANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

// #include <omp.h>

#include <fcntl.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unsupported/Eigen/KroneckerProduct>
//
#include <lattice/honeycomb.hpp>
#include <lattice/honeycombS3.hpp>
#include <lattice/honeycomb_hex.hpp>
#include <lattice/square.hpp>
#include <lattice/toric_lattice.hpp>
#include <machine/abstract_machine.hpp>
#include <machine/correlator.hpp>
#include <machine/file_psi.hpp>
#include <machine/pfaffian.hpp>
#include <machine/pfaffian_psi.hpp>
#include <machine/rbm_base.hpp>
#include <machine/rbm_symmetry.hpp>
#include <math.hpp>
#include <model/isingS3.hpp>
#include <model/kitaev.hpp>
#include <model/kitaevS3.hpp>
#include <model/toric.hpp>
#include <operators/aggregator.hpp>
#include <operators/bond_op.hpp>
#include <operators/derivative_op.hpp>
#include <operators/local_op.hpp>
#include <operators/local_op_chain.hpp>
#include <operators/overlap_op.hpp>
#include <operators/store_state.hpp>
#include <optimizer/abstract_optimizer.hpp>
#include <optimizer/gradient_descent.hpp>
#include <optimizer/minres_adapter.hpp>
#include <optimizer/plugin.hpp>
#include <optimizer/stochastic_reconfiguration.hpp>
#include <sampler/abstract_sampler.hpp>
#include <sampler/full_sampler.hpp>
#include <sampler/metropolis_sampler.hpp>
#include <tools/ini.hpp>
#include <tools/logger.hpp>
#include <tools/mpi.hpp>
#include <tools/state.hpp>
#include <tools/time_keeper.hpp>

#include "mpi.h"

using namespace Eigen;

volatile static bool g_interrupt = false;
volatile static bool g_saved = false;
int g_x;

void interrupt(int sig) {
    g_interrupt = sig == SIGINT;
    while (!g_saved) {
        usleep(100);
    }
    exit(0);
}

void progress_bar(size_t i, size_t n_epochs, double energy, char state) {
    double progress = i / (double)n_epochs;
    int digs = (int)std::log10(n_epochs) - (int)std::log10(i);
    if (i == 0) digs = (int)std::log10(n_epochs);
    std::cout << "\rEpochs: (" << std::string(digs, ' ') << i << "/" << n_epochs
              << ") " << state << " ";
    int x = g_x;
    int plen = x - (2 * (int)std::log10(n_epochs) + 37);
    int p = (int)(plen * progress + 0.5);
    int m = plen - p;
    std::cout << "[" << std::string(p, '#') << std::string(m, ' ') << "]";
    std::cout << std::showpoint;
    std::cout << " Energy: " << energy;
    std::cout << std::flush;
}

std::vector<size_t> to_indices(const MatrixXcd& vec) {
    std::vector<size_t> ret;
    for (size_t v = 0; v < (size_t)vec.size(); v++) {
        if (std::real(vec(v)) > 0) ret.push_back(v);
    }
    return ret;
}
size_t to_idx(const MatrixXcd& vec) {
    for (size_t v = 0; v < (size_t)vec.size(); v++) {
        if (std::real(vec(v)) > 0) return v;
    }
    return -1;
}
void test_symmetry() {
    model::kitaev km{4, -1};
    auto& lattice = km.get_lattice();
    MatrixXcd vec(lattice.n_total, 1);
    vec.setConstant(-1);
    vec(0) = 1;
    vec(1) = 1;
    vec(2) = 1;
    vec(8) = 1;

    auto symm = lattice.construct_symmetry();
    for (auto& s : symm) {
        lattice.print_lattice(to_indices(s * vec));
    }
}

void print_bonds() {
    lattice::honeycomb lat{2, 3};
    // lat.print_lattice({});
    auto bonds = lat.get_bonds();
    for (auto& bond : bonds) {
        std::cout << bond.a << "," << bond.b << "," << bond.type << std::endl;
    }
}

void test_S3() {
    model::isingS3 km{3, -1};
    machine::file_psi m{km.get_lattice(), "isingS3.state"};
    sampler::full_sampler sampler{m, 3};
    auto& h = km.get_hamiltonian();
    operators::aggregator agg{h};
    sampler.register_op(&h);
    sampler.register_agg(&agg);
    sampler.sample(false);
    std::cout.precision(17);
    std::cout << agg.get_result() << std::endl;
    std::cout << agg.get_result() / km.get_lattice().n_total << std::endl;
}

void debug() {
    //
    size_t n_chains = 16;
    size_t step_size = 5;
    size_t warmup_steps = 100;
    size_t n_samples = 1000;
    bool bond_flips = true;

    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};
    std::cout.precision(17);

    model::kitaev m{3, -1};
    machine::file_psi rbm{m.get_lattice(), "notebooks/n3.state"};
    sampler::full_sampler sampler{rbm, 3};
    sampler::metropolis_sampler msampler{
        rbm, n_samples, rng, n_chains, step_size, warmup_steps, bond_flips};
    operators::aggregator agg{m.get_hamiltonian()};
    agg.track_variance();
    sampler.register_op(&(m.get_hamiltonian()));
    sampler.register_agg(&agg);
    msampler.register_op(&(m.get_hamiltonian()));
    msampler.register_agg(&agg);

    for (size_t i = 0; i < 10; i++) {
        msampler.sample();
        std::cout << "Metropolis Sampler: " << agg.get_result() / rbm.n_visible
                  << " += " << agg.get_variance() / rbm.n_visible << std::endl;
        std::cout << msampler.get_acceptance_rate() << std::endl;
    }
    sampler.sample(false);
    std::cout << "Full Sampler: " << agg.get_result() / rbm.n_visible
              << " += " << agg.get_variance() / rbm.n_visible << std::endl;
}

Eigen::SparseMatrix<std::complex<double>> kron(
    const std::vector<Eigen::SparseMatrix<std::complex<double>>>& args) {
    Eigen::SparseMatrix<std::complex<double>> so(1, 1);
    so.insert(0, 0) = 1;
    for (const auto& arg : args) so = kroneckerProduct(arg, so).eval();
    return so;
}

void debug1() {
    using SparseXcd = Eigen::SparseMatrix<std::complex<double>>;
    std::cout << "DEBUG 1" << std::endl;
    SparseXcd sx(2, 2), sy(2, 2), sz(2, 2);
    sx.insert(0, 1) = 1;
    sx.insert(1, 0) = 1;
    sy.insert(0, 1) = std::complex<double>(0, -1);
    sy.insert(1, 0) = std::complex<double>(0, 1);
    sz.insert(0, 0) = 1;
    sz.insert(1, 1) = -1;

    SparseXcd x_yz = kron({sy, sx});
    SparseXcd x_zy = kron({sx, sy});

    SparseXcd y_xz = kron({-sy, sy});
    SparseXcd y_zx = kron({sy, -sy});

    SparseXcd z_xy = kron({sx, sx});
    SparseXcd z_yx = kron({sx, sx});
}

void debug_pfaffian() {
    lattice::honeycomb lat{2};
    machine::pfaffian pfaff{lat};
    Eigen::MatrixXcd state = Eigen::MatrixXd::Random(lat.n_total, 1);
    state.array() /= state.array().abs();

    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};
    pfaff.init_weights(rng, 0.1, false);

    auto context = pfaff.get_context(state);

    std::vector<size_t> flips;

    std::uniform_int_distribution<size_t> f_dist(0, lat.n_total - 1);
    Eigen::ArrayXd arr(pfaff.get_n_params(), 1);
    arr.setZero();
    for (size_t x = 0; x < 1000; x++) {
        flips.clear();
        for (size_t i = 0; i < 2; i++) {
            size_t r = f_dist(rng);
            if (std::find(flips.begin(), flips.end(), r) == flips.end()) {
                flips.push_back(r);
            }
        }
        pfaff.update_context(state, flips, context);
        for (auto& f : flips) state(f) *= -1;
        Eigen::MatrixXcd mat(pfaff.get_n_params(), 1);
        size_t o = 0;
        pfaff.derivative(state, context, mat, o);
        std::cout << (mat.array().abs() < 1e-10).cast<int>().sum() << std::endl;
        arr += (mat.array().abs() < 1e-10).cast<double>();
    }
    arr /= 1000.0;
    std::cout << arr.transpose() << std::endl;
    double mean = arr.mean();
    double stddev = std::sqrt(arr.square().mean() - std::pow(mean, 2));

    std::cout << mean << ", " << stddev << std::endl;
    std::cout << context.pfaff << " x10^" << context.exp << std::endl;

    auto context2 = pfaff.get_context(state);
    std::cout << context2.pfaff << " x10^" << context2.exp << std::endl;
    // Eigen::MatrixXcd mat = pfaff.get_mat(state).inverse();
    // std::cout << (context.inv - mat).norm() / mat.size() << std::endl;
}

void debug_pfaffian2() {
    lattice::honeycomb lat{2};
    machine::pfaffian pfaff{lat};
    Eigen::MatrixXcd state = Eigen::MatrixXd::Random(lat.n_total, 1);
    state.array() /= state.array().abs();

    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};
    pfaff.init_weights(rng, 0.1, false);

    auto context = pfaff.get_context(state);

    Eigen::MatrixXcd derivative(pfaff.get_n_params(), 1);
    size_t o = 0;
    pfaff.derivative(state, context, derivative, o);

    Eigen::MatrixXcd upd = Eigen::MatrixXcd::Random(pfaff.get_n_params(), 1);
    upd *= 1e-6;

    std::complex<double> pf1 = pfaff.psi(state, context);
    pf1 += pf1 * (derivative.array() * upd.array()).sum();
    std::cout.precision(17);
    std::cout << pf1 << std::endl;

    o = 0;
    pfaff.update_weights(upd, o);
    auto context2 = pfaff.get_context(state);
    std::complex<double> pf2 = pfaff.psi(state, context2);
    std::cout << pf2 << std::endl;

    std::cout << std::abs(pf1 - pf2) << std::endl;
}

void test_minresqlp() {
    int na = 500, nb = 500, nn = 50;
    Eigen::MatrixXcd mat(na, nb);
    double norm = 0.1;

    double e1 = 1;
    double de = 1;
    double e2 = 1;
    mat.setRandom();

    Eigen::MatrixXcd vec;
    vec = mat.rowwise().sum().conjugate() / std::sqrt(norm);

    MatrixXcd S(na, na);
    S = mat.conjugate() * mat.transpose() / norm;
    S -= vec * vec.transpose().conjugate();

    /* MatrixXcd d = mat.cwiseAbs2().rowwise().sum() / norm - vec.cwiseAbs2();
    std::cout << (S.diagonal() - d).norm() << std::endl; */

    double maxDiag = S.diagonal().real().maxCoeff();
    S.diagonal().topRows(nn) *= (1 + e1);
    S.diagonal().bottomRows(na - nn) *= (1 + e1 + de);
    S += e2 * maxDiag * Eigen::MatrixXcd::Identity(na, na);
    // S += e2 * Eigen::MatrixXcd::Identity(na, na);

    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> x(na), y(na);
    Eigen::VectorXcd z(na);
    x.setRandom();
    y.setZero();
    z.setZero();

    Eigen::VectorXcd tmp(nb, 1);
    Eigen::VectorXcd diag(na, 1);
    diag = mat.cwiseAbs2().rowwise().sum() / norm - vec.cwiseAbs2();
    optimizer::minresqlp_adapter min{mat, vec, e1, e2, de, norm, nn, diag, tmp};

    min.itnlim = 1000;
    std::cout << "start" << std::endl;
    std::cout << min.apply(x, z) << std::endl;
    std::cout << min.getItn() << std::endl;
    std::cout << min.getAcond() << std::endl;
    std::cout << min.getRnorm() << std::endl;

    y = S * z;

    std::cout << (x - y).norm() << std::endl;
}

void debug_general_pfaffprocedure() {
    size_t n = 1000, m = 10;
    MatrixXcd A(n, n);
    A.setRandom();
    A -= A.transpose().eval();
    A /= 2;

    MatrixXcd inv = A.inverse();
    inv -= inv.transpose().eval();
    inv /= 2;

    double diff = (MatrixXcd::Identity(n, n) - inv * A).array().abs().mean();
    std::cout << diff << std::endl;

    MatrixXcd Acopy = A;
    int expA;
    std::complex<double> pfaffA = math::pfaffian10(Acopy, expA);

    MatrixXcd B(n, m), C(m, m);
    B.setRandom();
    C.setRandom();
    C -= C.transpose().eval();
    C /= 2;
    MatrixXcd Cinv = C.inverse();
    Cinv -= Cinv.transpose().eval();
    Cinv /= 2;

    MatrixXcd BCB = B * C * B.transpose();
    MatrixXcd BinvB = B.transpose() * inv * B;

    MatrixXcd ABCB = A + BCB;
    MatrixXcd CinvBinvB = Cinv + BinvB;

    MatrixXcd ABCBcopy = ABCB;
    int expABCB;
    std::complex<double> pfaffABCB = math::pfaffian10(ABCBcopy, expABCB);

    MatrixXcd CinvBinvBcopy = CinvBinvB;
    int expCinvBinvB;
    std::complex<double> pfaffCinvBinvB =
        math::pfaffian10(CinvBinvBcopy, expCinvBinvB);

    MatrixXcd Cinvcopy = Cinv;
    int expCinv;
    std::complex<double> pfaffCinv = math::pfaffian10(Cinvcopy, expCinv);

    std::complex<double> pfaffABCB2 = pfaffA * pfaffCinvBinvB / pfaffCinv;

    int expABCB2 = expA + expCinvBinvB - expCinv;

    std::cout << pfaffABCB << " x10^" << expABCB << std::endl;
    std::cout << pfaffABCB2 << " x10^" << expABCB2 << std::endl;
    std::cout << pfaffA << " x10^" << expA << std::endl;
    std::cout << pfaffCinvBinvB << std::endl;
    std::cout << pfaffCinv << std::endl;
    std::cout << std::abs(pfaffABCB2 -
                          pfaffABCB * std::pow(10, expABCB - expABCB2))
              << std::endl;

    MatrixXcd ABCBinv = ABCB.inverse();
    ABCBinv -= ABCBinv.transpose().eval();
    ABCBinv /= 2;
    MatrixXcd ABCBinv2 = inv * B * CinvBinvB.inverse() * B.transpose() * inv;
    ABCBinv2 -= ABCBinv2.transpose().eval();
    ABCBinv2 /= 2;
    ABCBinv2 -= inv;

    std::cout << (ABCBinv + ABCBinv2).array().abs().mean() << std::endl;
}

void debugAprod() {
    int n = 100, m = 20;
    Eigen::MatrixXcd mat(n, m);
    mat.setRandom();
    Eigen::MatrixXcd S = mat.conjugate() * mat.transpose();
    Eigen::MatrixXcd x(n, 1);
    x.setRandom();
    Eigen::MatrixXcd y1(n, 1);
    Eigen::MatrixXcd y2(m, 1);
    Eigen::MatrixXcd tmp(m, 1);
    Eigen::MatrixXcd tmp2(m, 1);
    Eigen::MatrixXcd vec = Eigen::MatrixXcd::Zero(n, 1);
    std::complex<double> dot;
    Eigen::MatrixXcd diag = S.diagonal();

    double norm = 1.;
    double reg[] = {0., 0.};

    optimizer::g_mat = mat.data();
    optimizer::g_vec = vec.data();
    optimizer::g_tmp = tmp.data();
    optimizer::g_dot = &dot;
    optimizer::g_diag = diag.data();

    optimizer::g_norm = norm;
    optimizer::g_reg = &reg[0];

    optimizer::g_mat_dim2 = m;

    optimizer::Aprod(&n, x.data(), y1.data());
    tmp2 = mat.transpose() * x;
    y2 = S * x;
    Eigen::MatrixXcd r1(n, 1);
    Eigen::MatrixXcd r2(n, 1);
    optimizer::Aprod(&n, y1.data(), r1.data());
    r2 = S * y2;

    std::cout << y1.squaredNorm() << std::endl;
    std::cout << r1.adjoint() * x << std::endl;

    // std::cout << (y1 - y2).cwiseAbs2().mean() << std::endl;
    // std::cout << (tmp2 - tmp).cwiseAbs2().mean() << std::endl;
}

void init_seed(size_t g_seed, std::unique_ptr<std::mt19937>& rng) {
    std::uniform_int_distribution<unsigned long> udist{0, ULONG_MAX};

    unsigned long seed;
    if (mpi::master) {
        rng = std::make_unique<std::mt19937>(
            static_cast<std::mt19937::result_type>(g_seed));
        for (int i = 1; i < mpi::n_proc; i++) {
            seed = udist(*rng);
            MPI_Send(&seed, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&seed, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    if (!mpi::master)
        rng = std::make_unique<std::mt19937>(
            static_cast<std::mt19937::result_type>(seed));
}

int init_model(std::unique_ptr<model::abstract_model>& model) {
    switch (ini::model) {
        case ini::model_t::KITAEV:
            model = std::make_unique<model::kitaev>(
                ini::n_cells, ini::J.strengths, ini::n_cells_b, ini::full_symm,
                ini::lattice_type == "hex");
            break;
        case ini::model_t::KITAEV_S3:
            model = std::make_unique<model::kitaevS3>(ini::n_cells,
                                                      ini::J.strengths);
            break;
        case ini::model_t::ISING_S3:
            model = std::make_unique<model::isingS3>(ini::n_cells,
                                                     ini::J.strengths);
            break;
        case ini::model_t::TORIC:
            model =
                std::make_unique<model::toric>(ini::n_cells, ini::J.strengths);
            break;
        default:
            return 1;
    }
    if (model->supports_helper_hamiltonian() && ini::helper_strength != 0.) {
        model->add_helper_hamiltonian(ini::helper_strength);
    }
    return 0;
}

int init_machine(std::unique_ptr<machine::abstract_machine>& rbm,
                 machine::pfaffian* pfaff,
                 const std::unique_ptr<model::abstract_model>& model) {
    switch (ini::rbm) {
        case ini::rbm_t::BASIC:
            rbm = std::make_unique<machine::rbm_base>(
                ini::alpha, model->get_lattice(), ini::rbm_pop_mode,
                ini::rbm_cosh_mode);
            break;
        case ini::rbm_t::SYMMETRY:
            rbm = std::make_unique<machine::rbm_symmetry>(
                ini::alpha, model->get_lattice(), ini::rbm_pop_mode,
                ini::rbm_cosh_mode);
            break;
        case ini::rbm_t::FILE:
            rbm = std::make_unique<machine::file_psi>(model->get_lattice(),
                                                      ini::rbm_file_name);
            break;
        case ini::rbm_t::PFAFFIAN:
            rbm = std::make_unique<machine::pfaffian_psi>(model->get_lattice());
            break;
        default:
            return 2;
    }

    if (ini::rbm_correlators && model->get_lattice().has_correlators()) {
        auto c = model->get_lattice().get_correlators();
        rbm->add_correlators(c);
    }
    if (ini::rbm_pfaffian || ini::rbm == ini::rbm_t::PFAFFIAN) {
        pfaff = rbm->add_pfaffian(ini::rbm_pfaffian_symmetry).get();
    }
    return 0;
}

void init_weights(std::unique_ptr<machine::abstract_machine>& rbm,
                  machine::pfaffian* pfaff, bool force, std::mt19937& rng) {
    if (force || !rbm->load(ini::name)) {
        rbm->initialize_weights(rng, ini::rbm_weights, ini::rbm_weights_imag,
                                ini::rbm_weights_init_type);
        if (pfaff) {
            if (ini::rbm_pfaffian_load.empty() ||
                !pfaff->load_from_pfaffian_psi(ini::rbm_pfaffian_load)) {
                pfaff->init_weights(rng, ini::rbm_pfaffian_weights,
                                    ini::rbm_pfaffian_normalize);
            }
        }
    }
}

int init_sampler(std::unique_ptr<sampler::abstract_sampler>& sampler,
                 const std::unique_ptr<machine::abstract_machine>& rbm,
                 std::mt19937& rng) {
    switch (ini::sa_type) {
        case ini::sampler_t::FULL:
            sampler = std::make_unique<sampler::full_sampler>(
                *rbm, ini::sa_full_n_parallel_bits);
            break;
        case ini::sampler_t::METROPOLIS:
            sampler = std::make_unique<sampler::metropolis_sampler>(
                *rbm, ini::sa_n_samples, rng, ini::sa_metropolis_n_chains,
                ini::sa_metropolis_n_steps_per_sample,
                ini::sa_metropolis_n_warmup_steps,
                ini::sa_metropolis_bond_flips);
            break;
        default:
            return 4;
    }
    return 0;
}

int init_optimizer(std::unique_ptr<optimizer::abstract_optimizer>& optimizer,
                   std::unique_ptr<model::abstract_model>& model,
                   std::unique_ptr<machine::abstract_machine>& rbm,
                   std::unique_ptr<sampler::abstract_sampler>& sampler) {
    switch (ini::opt_type) {
        case ini::optimizer_t::SR:
            optimizer = std::make_unique<optimizer::stochastic_reconfiguration>(
                *rbm, *sampler, model->get_hamiltonian(), ini::opt_lr,
                ini::opt_sr_reg1, ini::opt_sr_reg2, ini::opt_sr_deltareg1,
                ini::opt_sr_method, ini::opt_sr_max_iterations,
                ini::opt_sr_rtol, ini::opt_resample, ini::opt_resample_alpha1,
                ini::opt_resample_alpha2, ini::opt_resample_alpha3);
            break;
        case ini::optimizer_t::SGD:
            optimizer = std::make_unique<optimizer::gradient_descent>(
                *rbm, *sampler, model->get_hamiltonian(), ini::opt_lr,
                ini::opt_sgd_real_factor, ini::opt_resample,
                ini::opt_resample_alpha1, ini::opt_resample_alpha2,
                ini::opt_resample_alpha3);
            break;
        default:
            return 8;
    }

    optimizer->register_observables();
    std::unique_ptr<optimizer::base_plugin> p;
    if (ini ::opt_plugin.length() > 0) {
        if (ini::opt_plugin == "adam") {
            p = std::make_unique<optimizer::adam_plugin>(
                rbm->get_n_params(), ini::opt_adam_beta1, ini::opt_adam_beta2,
                ini::opt_adam_eps);
        } else if (ini::opt_plugin == "momentum") {
            p = std::make_unique<optimizer::momentum_plugin>(
                rbm->get_n_params());
        } else if (ini::opt_plugin == "heun") {
            p = std::make_unique<optimizer::heun_plugin>(
                [&optimizer]() -> Eigen::VectorXcd& {
                    return optimizer->gradient(false);
                },
                *rbm, *sampler, ini::opt_heun_eps);
        } else {
            return 16;
        }
        optimizer->set_plugin(p.get());
    }
    return 0;
}

int main(int argc, char* argv[]) {
    // test_minresqlp();
    // return 0;
    //
    mpi::init(argc, argv);
    int rc = ini::load(argc, argv);
    if (rc != 0) {
        return rc;
    }

    if (ini::print_bonds && mpi::master) {
        std::unique_ptr<model::abstract_model> model;
        init_model(model);
        auto bonds = model->get_lattice().get_bonds();
        for (const auto& b : bonds) {
            std::cout << b.a << "," << b.b << "," << b.type << std::endl;
        }
        mpi::end();
        return 0;
    }

    if (mpi::master) {
        logger::init();
        std::cout << "Starting '" << ini::name << "'!" << std::endl;
    }

    omp_set_num_threads(2);
    std::unique_ptr<std::mt19937> rng;
    std::unique_ptr<model::abstract_model> model;
    std::unique_ptr<machine::abstract_machine> rbm;
    machine::pfaffian* pfaff = 0;
    std::unique_ptr<sampler::abstract_sampler> sampler;
    std::unique_ptr<optimizer::abstract_optimizer> optimizer;

    size_t seed = ini::seed;
    // Init Model
    rc |= init_model(model);
    // Init RBM
    rc |= init_machine(rbm, pfaff, model);

    if (ini::train && ini::seed_search && ini::rbm_force) {
        size_t best_seed;
        double best_energy = DBL_MAX;
        std::unique_ptr<std::mt19937> best_rng;
        std::unique_ptr<machine::abstract_machine> best_rbm;
        std::unique_ptr<sampler::abstract_sampler> best_sampler;
        std::unique_ptr<optimizer::abstract_optimizer> best_optimizer;
        std::uniform_int_distribution<unsigned long> udist(0, ULONG_MAX);
        for (int i = 0; i < ini::seed_search; i++) {
            init_seed(seed, rng);
            seed = udist(*rng);
            if (mpi::master) std::cout << "Seed: " << seed << std::endl;
            init_machine(rbm, pfaff, model);
            // Init Weights
            init_weights(rbm, pfaff, true, *rng);
            // Init Sampler
            rc |= init_sampler(sampler, rbm, *rng);
            rc |= init_optimizer(optimizer, model, rbm, sampler);

            for (size_t e = 0; e < ini::seed_search_epochs; e++) {
                sampler->sample();
                optimizer->optimize();
                logger::newline();
            }
            if (optimizer->get_current_energy() < best_energy) {
                best_energy = optimizer->get_current_energy();
                best_rng = std::move(rng);
                best_rbm = std::move(rbm);
                best_sampler = std::move(sampler);
                best_optimizer = std::move(optimizer);
                best_seed = seed;
            }

            rng.reset(0);
            rbm.reset(0);
            sampler.reset(0);
            optimizer.reset(0);
        }
        if(mpi::master)
            std::cout << "Best Seed: " << best_seed
                  << " at E=" << best_energy / best_rbm->n_visible << std::endl;
        seed = best_seed;
        rng = std::move(best_rng);
        rbm = std::move(best_rbm);
        sampler = std::move(best_sampler);
        optimizer = std::move(best_optimizer);
    } else {
        init_seed(seed, rng);
        if (mpi::master) std::cout << "Seed: " << ini::seed << std::endl;
        // Init Weights
        init_weights(rbm, pfaff, ini::rbm_force, *rng);
        // Init Sampler
        rc |= init_sampler(sampler, rbm, *rng);
        if (ini::train) {
            rc |= init_optimizer(optimizer, model, rbm, sampler);
        }
    }

    if (mpi::master) {
        std::cout << "Number of parameters: " << rbm->get_n_params()
                  << std::endl;
    }

    if (ini::train) {
        struct termios oldt, newt;
        int oldf;
        if (!ini::noprogress && mpi::master) {
            struct winsize size;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
            g_x = size.ws_col;
            if(g_x <= 0 || g_x > 500) {
                g_x = 100;
            }

            // Start getchar non-block
            tcgetattr(STDIN_FILENO, &oldt);
            newt = oldt;
            newt.c_lflag &= ~(ICANON | ECHO);
            tcsetattr(STDIN_FILENO, TCSANOW, &newt);
            oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
            fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
            // End getchar non-block
            progress_bar(0, ini::n_epochs, -1, 'S');
        }

        int ch = 0;
        for (int i = rbm->get_n_updates(); i < ini::n_epochs && ch != ''; i++) {
            if (i > 0 && i % 100 == 0) rbm->save(ini::name, true);
            time_keeper::start("Sampling");
            sampler->sample();
            time_keeper::end("Sampling");
            if (!ini::noprogress && mpi::master)
                progress_bar(i + 1, ini::n_epochs,
                             optimizer->get_current_energy() / rbm->n_visible,
                             'O');
            time_keeper::start("Optimization");
            optimizer->optimize();
            time_keeper::end("Optimization");
            if (std::isnan(optimizer->get_current_energy())) {
                std::cerr << "\nEnergy went NaN." << std::endl;
                rc = 55;
                break;
            }
            logger::newline();
            if (!ini::noprogress && mpi::master) {
                progress_bar(i + 1, ini::n_epochs,
                             optimizer->get_current_energy() / rbm->n_visible,
                             'S');
                ch = getchar();
            }
            time_keeper::itn();
        }
        if (mpi::master) time_keeper::resumee();

        if (!ini::noprogress && mpi::master) {
            // Start getchar non-block
            tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
            fcntl(STDIN_FILENO, F_SETFL, oldf);
            // End getchar non-block
            std::cout << std::endl;
        }

        // std::cout << optimizer->get_total_resamples() << std::endl;

        rbm->save(ini::name);
    }
    if (ini::evaluate) {
        sampler->clear_ops();
        sampler->clear_aggs();
        if (ini::sa_eval_samples != 0)
            sampler->set_n_samples(ini::sa_eval_samples);
        model::SparseXcd plaq_op =
            model::kron({model::sx(), model::sy(), model::sz(), model::sx(),
                         model::sy(), model::sz()});
        auto hex = dynamic_cast<lattice::honeycomb*>(&model->get_lattice())
                       ->get_hexagons();
        std::vector<operators::aggregator*> aggs;
        std::vector<operators::base_op*> ops;
        for (auto& h : hex) {
            // for (auto& x : h) std::cout << x << ", ";
            // std::cout << std::endl;

            operators::local_op* op = new operators::local_op(h, plaq_op);
            operators::aggregator* agg = new operators::aggregator(*op);
            ops.push_back(op);
            aggs.push_back(agg);
        }

        sampler->register_ops(ops);
        sampler->register_aggs(aggs);

        model->remove_helper_hamiltoian();
        auto& h = model->get_hamiltonian();
        operators::aggregator ah(h);
        ah.track_variance();
        sampler->register_op(&h);
        sampler->register_agg(&ah);

        sampler->sample();

        if (mpi::master) {
            std::cout << "BEGIN OUTPUT" << std::endl;
            std::cout.precision(16);
            for (size_t i = 0; i < hex.size(); i++) {
                std::cout << aggs[i]->get_result() << std::endl;
            }

            std::cout << ah.get_result() / rbm->n_visible << std::endl;
            std::cout << std::sqrt(ah.get_variance()(0)) / rbm->n_visible
                      << std::endl;
            if (ini::sa_type == ini::sampler_t::METROPOLIS) {
                std::cout << dynamic_cast<sampler::metropolis_sampler*>(
                                 sampler.get())
                                 ->get_acceptance_rate()
                          << std::endl;
            }
            std::cout << "END OUTPUT" << std::endl;
        }
    }

    // std::ofstream ws{"weights/weights_" + ini::name + ".txt"};
    // ws << "# Weights\n";
    // ws << rbm->get_weights();
    // ws << "\n\n# Hidden Bias\n";
    // ws << rbm->get_h_bias();
    // ws << "\n\n# Visible Bias\n";
    // ws << rbm->get_v_bias();
    // ws.close();

    // model->remove_helper_hamiltoian();
    // machine::full_sampler samp{*rbm, 3};
    // operators::aggregator agg{model->get_hamiltonian()};
    // samp.register_op(&(model->get_hamiltonian()));
    // samp.register_agg(&agg);
    // samp.sample(true);
    // std::cout.precision(17);
    // std::cout << std::real(agg.get_result()(0)) / rbm->n_visible <<
    // std::endl;
    //
    mpi::end();

    return rc;
}
