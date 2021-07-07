/*
 * Copyright (c) 2021 David Bucher <David.Bucher@physik.lmu.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
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
#include <omp.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
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
#include <lattice/toric_lattice.hpp>
#include <machine/abstract_machine.hpp>
#include <machine/correlator.hpp>
#include <machine/file_psi.hpp>
#include <machine/pfaffian.hpp>
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
#include <tools/state.hpp>

using namespace Eigen;

volatile static bool g_interrupt = false;
volatile static bool g_saved = false;

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
    struct winsize size;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
    int plen = size.ws_col - (2 * (int)std::log10(n_epochs) + 37);
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
    lattice::honeycomb lat{3};
    auto bonds = lat.get_bonds();
    for (auto& bond : bonds) {
        std::cout << bond.a << "," << bond.b << "," << bond.type << std::endl;
    }
    // lat.print_lattice({0, lat.up(0, 1)});
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
    lattice::honeycomb lat{8};
    machine::pfaffian pfaff{lat, 4};
    Eigen::MatrixXcd state = Eigen::MatrixXd::Random(lat.n_total, 1);
    state.array() /= state.array().abs();

    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};
    pfaff.init_weights(rng, 0.1, false);

    auto context = pfaff.get_context(state);

    std::vector<size_t> flips;

    std::uniform_int_distribution<size_t> f_dist(0, lat.n_total - 1);
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
    }
    std::cout << context.pfaff << " x10^" << context.exp << std::endl;

    auto context2 = pfaff.get_context(state);
    std::cout << context2.pfaff << " x10^" << context2.exp << std::endl;
    // Eigen::MatrixXcd mat = pfaff.get_mat(state).inverse();
    // std::cout << (context.inv - mat).norm() / mat.size() << std::endl;
}

void test_minresqlp() {
    int na = 10000, nb = 500, nn = 300;
    Eigen::MatrixXcd mat(na, nb);
    double norm = 0.1;

    double e1 = 0.1;
    double de = 0.1;
    double e2 = 0.1;
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
    Eigen::MatrixXcd z(na, 1);
    x.setRandom();
    x.normalize();
    y.setZero();
    z.setZero();

    // y = S.inverse() * x;
    // y.array() += d.array() * x.array() * e1;

    optimizer::minresqlp_adapter min{mat, vec, e1, e2, de, norm, nn};

    min.itnlim = 50;
    std::cout << "start" << std::endl;
    std::cout << min.apply(x, z) << std::endl;
    std::cout << min.getItn() << std::endl;
    std::cout << min.getAcond() << std::endl;
    std::cout << min.getRnorm() << std::endl;

    y = S * z;
    // y.normalize();

    std::cout << std::pow(std::abs(x.dot(y)), 2) << std::endl;
}

int main(int argc, char* argv[]) {
    int rc = ini::load(argc, argv);
    if (rc != 0) {
        return rc;
    }

    logger::init();

    omp_set_num_threads(ini::n_threads);
    Eigen::setNbThreads(1);

    std::cout << "Seed: " << ini::seed << std::endl;
    std::mt19937 rng{static_cast<std::mt19937::result_type>(ini::seed)};

    std::unique_ptr<model::abstract_model> model;
    switch (ini::model) {
        case ini::model_t::KITAEV:
            model = std::make_unique<model::kitaev>(ini::n_cells, ini::J,
                                                    ini::n_cells_b);
            break;
        case ini::model_t::KITAEV_S3:
            model = std::make_unique<model::kitaevS3>(ini::n_cells, ini::J);
            break;
        case ini::model_t::ISING_S3:
            model = std::make_unique<model::isingS3>(ini::n_cells, ini::J);
            break;
        case ini::model_t::TORIC:
            model = std::make_unique<model::toric>(ini::n_cells, ini::J);
            break;
        default:
            return 1;
    }
    if (model->supports_helper_hamiltonian() && ini::helper_strength != 0.) {
        model->add_helper_hamiltonian(ini::helper_strength);
    }

    std::unique_ptr<machine::abstract_machine> rbm;

    size_t n_hidden = ini::n_hidden;
    if (ini::alpha > 0.) {
        n_hidden =
            (size_t)std::round(ini::alpha * model->get_lattice().n_total);
    }
    std::cout << "Number of hidden units: " << n_hidden << std::endl;

    switch (ini::rbm) {
        case ini::rbm_t::BASIC:
            rbm = std::make_unique<machine::rbm_base>(
                n_hidden, model->get_lattice(), ini::rbm_pop_mode,
                ini::rbm_cosh_mode);
            break;
        case ini::rbm_t::SYMMETRY:
            rbm = std::make_unique<machine::rbm_symmetry>(
                n_hidden, model->get_lattice(), ini::rbm_pop_mode,
                ini::rbm_cosh_mode);
            break;
        // case ini::rbm_t::PFAFFIAN:
        //     rbm =
        //     std::make_unique<machine::pfaffian_psi>(model->get_lattice());
        //     break;
        default:
            return 1;
    }

    if (ini::rbm_correlators && model->get_lattice().has_correlators()) {
        auto c = model->get_lattice().get_correlators();
        rbm->add_correlators(c);
    }

    machine::pfaffian* pfaff = 0;
    if (ini::rbm_pfaffian || ini::rbm == ini::rbm_t::PFAFFIAN) {
        pfaff = rbm->add_pfaffian(ini::rbm_pfaffian_symmetry).get();
    }
    if (ini::rbm_force || !rbm->load(ini::name)) {
        rbm->initialize_weights(rng, ini::rbm_weights, ini::rbm_weights_imag,
                                ini::rbm_weights_init_type);
        if (pfaff)
            pfaff->init_weights(rng, ini::rbm_pfaffian_weights,
                                ini::rbm_pfaffian_normalize);
    }

    std::unique_ptr<sampler::abstract_sampler> sampler;
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
            return 1;
    }

    std::unique_ptr<optimizer::abstract_optimizer> optimizer;
    switch (ini::opt_type) {
        case ini::optimizer_t::SR:
            optimizer = std::make_unique<optimizer::stochastic_reconfiguration>(
                *rbm, *sampler, model->get_hamiltonian(), ini::opt_lr,
                ini::opt_sr_reg1, ini::opt_sr_reg2, ini::opt_sr_deltareg1,
                ini::opt_sr_iterative, ini::opt_sr_max_iterations,
                ini::opt_sr_rtol);
            break;
        case ini::optimizer_t::SGD:
            optimizer = std::make_unique<optimizer::gradient_descent>(
                *rbm, *sampler, model->get_hamiltonian(), ini::opt_lr,
                ini::opt_sgd_real_factor);
            break;
        default:
            return 1;
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
        } else {
            return 1;
        }
        optimizer->set_plugin(p.get());
    }

    std::cout << "Number of parameters: " << rbm->get_n_params() << std::endl;

    if (ini::train) {
        // Start getchar non-block
        struct termios oldt, newt;
        int oldf;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
        // End getchar non-block

        progress_bar(0, ini::n_epochs, -1, 'S');
        int ch = 0;
        for (size_t i = 0; i < ini::n_epochs && ch != ''; i++) {
            Eigen::setNbThreads(1);
            sampler->sample();
            progress_bar(i + 1, ini::n_epochs,
                         optimizer->get_current_energy() / rbm->n_visible, 'O');
            Eigen::setNbThreads(-1);
            optimizer->optimize();
            logger::newline();
            progress_bar(i + 1, ini::n_epochs,
                         optimizer->get_current_energy() / rbm->n_visible, 'S');
            ch = getchar();
        }

        // Start getchar non-block
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        fcntl(STDIN_FILENO, F_SETFL, oldf);
        // End getchar non-block

        std::cout << std::endl;
        rbm->save(ini::name);

    } else {
        std::cout << "nothing t do!" << std::endl;
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

    return 0;
}
