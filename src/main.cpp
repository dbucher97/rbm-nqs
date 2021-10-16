/*
 * Copyright (c) 2021 David Bucher <David.Bucher@physik.lmu.de> * * This program
 * is free software: you can redistribute it and/or modify * it under the terms
 * of the GNU Affero General Public License as * published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
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

#include <fcntl.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
//
#include <mpi.h>

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
#include <model/isingS3.hpp>
#include <model/kitaev.hpp>
#include <model/kitaevS3.hpp>
#include <model/toric.hpp>
#include <operators/store_state.hpp>
#include <optimizer/abstract_optimizer.hpp>
#include <optimizer/gradient_descent.hpp>
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

using namespace Eigen;

int g_x;

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
                ini::n_cells, ini::J.strengths, ini::n_cells_b,
                ini::symmetry.symm, ini::lattice_type == "hex");
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
                 machine::pfaffian*& pfaff,
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
        pfaff = rbm->add_pfaffian(ini::rbm_pfaffian_symmetry.symm,
                                  ini::rbm_pfaffian_no_updating)
                    .get();
    }
    return 0;
}

void init_weights(std::unique_ptr<machine::abstract_machine>& rbm,
                  machine::pfaffian* pfaff,
                  const std::unique_ptr<model::abstract_model>& model,
                  bool force, std::mt19937& rng) {
    if (force || !rbm->load(ini::name)) {
        rbm->initialize_weights(rng, ini::rbm_weights, ini::rbm_weights_imag,
                                ini::rbm_weights_init_type);
        if (pfaff) {
            if (ini::rbm_pfaffian_load.empty() ||
                !pfaff->load_from_pfaffian_psi(ini::rbm_pfaffian_load)) {
                pfaff->init_weights(rng, ini::rbm_pfaffian_weights,
                                    ini::rbm_pfaffian_normalize);
            }
            // std::vector<Eigen::SparseMatrix<std::complex<double>>> mats;
            // std::vector<std::vector<size_t>> acts_on;

            // for (auto& op : model->get_hamiltonian().get_ops()) {
            //     mats.push_back(
            //         dynamic_cast<operators::local_op*>(op)->get_op());
            //     acts_on.push_back(
            //         dynamic_cast<operators::local_op*>(op)->get_acts_on());
            // }

            // pfaff->init_weights_hf(mats, acts_on);
        }
    }
}

int init_sampler(std::unique_ptr<sampler::abstract_sampler>& sampler,
                 const std::unique_ptr<machine::abstract_machine>& rbm,
                 std::mt19937& rng) {
    switch (ini::sa_type) {
        case ini::sampler_t::FULL:
            sampler = std::make_unique<sampler::full_sampler>(
                *rbm, ini::sa_full_n_parallel_bits, ini::sa_pfaffian_refresh,
                ini::sa_lut_exchange);
            break;
        case ini::sampler_t::METROPOLIS:
            sampler = std::make_unique<sampler::metropolis_sampler>(
                *rbm, ini::sa_n_samples, rng, ini::sa_metropolis_n_chains,
                ini::sa_metropolis_n_steps_per_sample,
                ini::sa_metropolis_n_warmup_steps,
                ini::sa_metropolis_bond_flips, ini::sa_pfaffian_refresh,
                ini::sa_lut_exchange);
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

void store_state(std::unique_ptr<model::abstract_model>& model,
                 std::unique_ptr<machine::abstract_machine>& rbm,
                 machine::pfaffian* pfaff, std::mt19937& rng) {
    init_machine(rbm, pfaff, model);
    init_weights(rbm, pfaff, model, false, rng);
    sampler::full_sampler sampler{*rbm, ini::sa_full_n_parallel_bits};
    mpi::cout << "Storing State..." << mpi::endl;
    sampler.sample(true);
}

void store_samples(std::unique_ptr<sampler::abstract_sampler>& sampler) {
    operators::store_state ss{ini::name + ".samples"};
    sampler->register_op(&ss);
    sampler->sample();
}

void debug_() {
    lattice::honeycomb_hex l(2);
    std::vector<double> symm = {2};
    auto sp = l.construct_symmetry(symm);
    auto sb = l.construct_symm_basis(symm);

    for (size_t i = 0; i < sp.size(); i++) {
        Eigen::MatrixXcd state(l.n_total, 1);
        state.setOnes();
        for (auto& b : sb) state(b) = -1;

        state = sp[i] * state;

        std::vector<size_t> x;
        for (int i = 0; i < state.size(); i++) {
            if (std::real(state(i)) < 0) x.push_back(i);
        }
        l.print_lattice(x);
    }
}

int main(int argc, char* argv[]) {
    mpi::init(argc, argv);
    int rc = ini::load(argc, argv);
    if (rc != 0) {
        return rc;
    }

    if (ini::print_bonds && mpi::master) {
        std::unique_ptr<model::abstract_model> model;
        init_model(model);
        if (ini::model == ini::TORIC) {
            auto plaq =
                dynamic_cast<lattice::toric_lattice*>(&model->get_lattice())
                    ->construct_plaqs();
            for (auto& p : plaq) {
                for (auto& i : p.idxs) std::cout << i << ",";
                std::cout << p.type << std::endl;
            }
        } else {
            auto bonds = model->get_lattice().get_bonds();
            for (const auto& b : bonds) {
                std::cout << b.a << "," << b.b << "," << b.type << std::endl;
            }
        }
        mpi::end();
        return 0;
    }

    if (ini::print_hex && mpi::master && ini::model == ini::KITAEV) {
        std::unique_ptr<model::abstract_model> model;
        init_model(model);
        auto hex = dynamic_cast<lattice::honeycomb*>(&model->get_lattice())
                       ->get_hexagons();
        for (const auto& h : hex) {
            for (const auto& x : h) std::cout << x << ", ";
            std::cout << std::endl;
        }
        mpi::end();
        return 0;
    }

    if (mpi::master) {
        logger::init();
    }
    mpi::cout << "Starting '" << ini::name << "'!" << mpi::endl;

    std::unique_ptr<std::mt19937> rng;
    std::unique_ptr<model::abstract_model> model;
    std::unique_ptr<machine::abstract_machine> rbm;
    machine::pfaffian* pfaff = 0;
    std::unique_ptr<sampler::abstract_sampler> sampler;
    std::unique_ptr<optimizer::abstract_optimizer> optimizer;

    size_t seed = ini::seed;
    // Init Model
    rc |= init_model(model);

    if (ini::store_state) {
        store_state(model, rbm, pfaff, *rng);
        mpi::end();
        return 0;
    }

    /* auto x = model->get_lattice().construct_symmetry();
    if (mpi::master) {
        std::cout << "[";
        for (auto& s : x) {
            std::cout << "[";
            for (int i = 0; i < s.size(); i++) {
                std::cout << s.indices()(i) << ", ";
            }
            std::cout << "]," << std::endl;
        }
        std::cout << "]" << std::endl;
    }
    mpi::end();
    return 0; */

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
            mpi::cout << "Seed: " << seed << " \t" << mpi::flush;
            // Init RBM
            init_machine(rbm, pfaff, model);
            // Init Weights
            init_weights(rbm, pfaff, model, true, *rng);
            // Init Sampler
            rc |= init_sampler(sampler, rbm, *rng);
            rc |= init_optimizer(optimizer, model, rbm, sampler);

            double energy = 0;

            for (size_t e = 0; e < ini::seed_search_epochs; e++) {
                sampler->sample();
                optimizer->optimize();
                logger::newline();
                energy = 0.9 * energy + optimizer->get_current_energy();
                if (std::isnan(optimizer->get_current_energy())) break;
            }
            mpi::cout << energy * 0.1 / rbm->n_visible << mpi::endl;
            if (energy < best_energy) {
                best_energy = energy;
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
        if (best_energy == DBL_MAX) {
            rc = 65;
        } else {
            mpi::cout << "Best Seed: " << best_seed << " at E="
                      << best_optimizer->get_current_energy() /
                             best_rbm->n_visible
                      << mpi::endl;
        }
        seed = best_seed;
        rng = std::move(best_rng);
        rbm = std::move(best_rbm);
        sampler = std::move(best_sampler);
        optimizer = std::move(best_optimizer);
        time_keeper::clear();
    } else {
        init_seed(seed, rng);
        mpi::cout << "Seed: " << ini::seed << mpi::endl;
        // Init RBM
        init_machine(rbm, pfaff, model);
        // Init Weights
        init_weights(rbm, pfaff, model, ini::rbm_force, *rng);
        // Init Sampler
        rc |= init_sampler(sampler, rbm, *rng);
        if (ini::train) {
            rc |= init_optimizer(optimizer, model, rbm, sampler);
        }
    }

    if (ini::store_samples) {
        store_samples(sampler);
        mpi::end();
        return 0;
    }

    // if (mpi::master) {
    //     auto& l = model->get_lattice();
    //     std::vector<int> x = {
    //     for (size_t j = 0; j < x.size(); j++) {
    //         std::vector<size_t> ones;
    //         for (size_t i = 0; i < rbm->n_visible; i++) {
    //             if ((x[j] >> i) & 1) {
    //                 ones.push_back(i);
    //             }
    //         }
    //         l.print_lattice(ones);
    //     }
    // }
    // mpi::end();
    // return 0;

    if (ini::train && rc == 0) {
        mpi::cout << "Number of parameters: " << rbm->get_n_params()
                  << mpi::endl;
        struct termios oldt, newt;
        int oldf;
        if (!ini::noprogress && mpi::master) {
            struct winsize size;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
            g_x = size.ws_col;
            if (g_x <= 0 || g_x > 500) {
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
        for (int i = rbm->get_n_updates(); i < (int)ini::n_epochs && ch != '';
             i++) {
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
                if (mpi::master)
                    std::cerr << "\nFATAL: NaN Energy." << std::endl;
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
        time_keeper::resumee();

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
    if (ini::evaluate && rc == 0) {
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

        // sampler->register_ops(ops);
        // sampler->register_aggs(aggs);

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
            // for (size_t i = 0; i < hex.size(); i++) {
            //     std::cout << aggs[i]->get_result() << std::endl;
            // }

            std::cout << ah.get_result() / rbm->n_visible << std::endl;
            std::cout << ah.get_variance()(0) << std::endl;
            std::cout << std::sqrt(ah.get_variance()(0)) / rbm->n_visible
                      << std::endl;
            if (ini::sa_type == ini::sampler_t::METROPOLIS) {
                std::cout << dynamic_cast<sampler::metropolis_sampler*>(
                                 sampler.get())
                                 ->get_acceptance_rate()
                          << std::endl;
            }

            // std::cout << rbm->get_pfaffian().get_weights() << std::endl;
            std::cout << "END OUTPUT" << std::endl;
        }
    }

    auto samples = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (auto& s : samples) {
        sampler->clear_ops();
        sampler->clear_aggs();

        sampler->set_n_samples(s);

        auto& h = model->get_hamiltonian();
        operators::aggregator ah(h);
        ah.track_variance();
        sampler->register_op(&h);
        sampler->register_agg(&ah);

        sampler->sample();

        std::cout << s << ", ";
        std::cout << std::real(ah.get_result()(0)) / rbm->n_visible << ", ";
        std::cout << ah.get_variance()(0) / rbm->n_visible << std::endl;
    }

    mpi::end();

    return rc;
}
