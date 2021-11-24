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
#include <iomanip>
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

#include "operators/aggregator.hpp"

using namespace Eigen;

int g_x;

void progress_bar(size_t i, size_t n_epochs, double energy, char state) {
    double progress = i / (double)n_epochs;
    std::cout.precision(5);
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
    if (!mpi::master) {
        rng = std::make_unique<std::mt19937>(
            static_cast<std::mt19937::result_type>(seed));
    }
}

int init_model(std::unique_ptr<model::abstract_model>& model) {
    switch (ini::model) {
        case ini::model_t::KITAEV:
            model = std::make_unique<model::kitaev>(
                ini::n_cells, ini::J.strengths, ini::n_cells_b,
                ini::symmetry.symm, ini::lattice_type == "hex", ini::h);
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
                   std::unique_ptr<sampler::abstract_sampler>& sampler,
                   std::unique_ptr<optimizer::base_plugin>& plug) {
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
    if (ini ::opt_plugin.length() > 0) {
        if (ini::opt_plugin == "adam") {
            plug = std::make_unique<optimizer::adam_plugin>(
                rbm->get_n_params(), ini::opt_adam_beta1, ini::opt_adam_beta2,
                ini::opt_adam_eps);
        } else if (ini::opt_plugin == "momentum") {
            plug = std::make_unique<optimizer::momentum_plugin>(
                rbm->get_n_params());
        } else if (ini::opt_plugin == "heun") {
            plug = std::make_unique<optimizer::heun_plugin>(
                [&optimizer]() -> Eigen::VectorXcd& {
                    return optimizer->gradient(false);
                },
                *rbm, *sampler, ini::opt_heun_eps);
        } else {
            return 16;
        }
        optimizer->set_plugin(plug.get());
    }
    return 0;
}

void store_state() {
    std::unique_ptr<std::mt19937> rng;
    std::unique_ptr<model::abstract_model> model;
    std::unique_ptr<machine::abstract_machine> rbm;
    machine::pfaffian* pfaff = 0;
    init_seed(ini::seed, rng);
    init_model(model);
    init_machine(rbm, pfaff, model);
    init_weights(rbm, pfaff, model, ini::rbm_force, *rng);

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
    lattice::honeycomb l(4);
    std::vector<double> symm = {1};
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
        std::cout << "\n\n";
    }
}

class mpi_op : public operators::base_op {
    int c;
    std::vector<double> data = {
        0.977032588723069,    0.042131676446022226, 0.522972009002048,
        0.9421396560875159,   0.6596102035244401,   0.19811722085692818,
        0.1946833881000004,   0.8062301027565533,   0.6562266325318439,
        0.48292217432207507,  0.7721003971742948,   0.6185413881658379,
        0.5921020910720973,   0.30968682631762035,  0.811837887251313,
        0.6541725158631927,   0.7147620088231902,   0.4647773268829073,
        0.28871355708785296,  0.16100667185583595,  0.22367624027619448,
        0.9260344138408682,   0.11971591939205839,  0.5485936384659809,
        0.24020063756596544,  0.9951943223262818,   0.11871793538892472,
        0.32819109031512095,  0.061712317868371724, 0.10710638346252987,
        0.38592808271501655,  0.47642536016437664,  0.9544768504764556,
        0.11149792063334596,  0.48076428732587784,  0.728971578762767,
        0.6755948519482451,   0.13594778958574438,  0.9157750018193435,
        0.803026551955031,    0.04552700338411331,  0.4549025242241056,
        0.25592729780108936,  0.22033681586531284,  0.5195658825402238,
        0.5497213952170441,   0.42090051466777356,  0.7916682122492474,
        0.9174124189479687,   0.7586731879556448,   0.15978385054534705,
        0.8118136128697562,   0.2638834080579403,   0.6069316133322737,
        0.491747117083838,    0.117078225042749,    0.9917580773865583,
        0.8698788302327972,   0.8251456209145848,   0.9598906838075068,
        0.44940365307257446,  0.3155972426486876,   0.3782726410028242,
        0.961742356202122,    0.7925214329259596,   0.21307731725958945,
        0.3808289524565299,   0.2913315607594972,   0.3950034867105411,
        0.14472427506489172,  0.8592815554444933,   0.1279657613256472,
        0.5899733772656922,   0.07756903222525457,  0.6575053800445048,
        0.391748644113212,    0.12309910967999027,  0.6558721866060615,
        0.038962162027456504, 0.6739260665406597,   0.49481251179803243,
        0.6777140352879767,   0.5681041828859267,   0.2557817404081909,
        0.11198020897180305,  0.6663278185460747,   0.4670812764966634,
        0.6005785862735303,   0.06722373694535444,  0.04205412612252779,
        0.3864607903849878,   0.030456075790041837, 0.13415789484902774,
        0.5002084338197285,   0.7433096579216042,   0.34077171216312485,
        0.36287913750323175,  0.29221880899062425,  0.19630683936038662,
        0.5406507921681055};

    std::vector<double> ps = {
        0.010875994302703739,  0.010594101628117553,   0.01283932787743003,
        0.0012179711727699453, 0.013614815770084478,   0.015374538681990011,
        0.006336413164670813,  0.01482163500000825,    0.015134265262609215,
        0.010249964996904878,  0.006840953286031941,   0.01690020902195116,
        0.0009470325045802447, 0.0003382585619616578,  0.008548569561994079,
        0.015092702985082408,  0.011472397587385837,   0.008445139007449006,
        0.003740274628480024,  0.003510884282858516,   0.009866532736965814,
        0.01776655181566469,   0.007939448767547683,   0.0018341151952600164,
        0.015310232235090447,  0.017486891697752516,   0.005311548549953812,
        0.014887949024443384,  0.004738711168914282,   0.009540912158783504,
        0.0063935012181789595, 0.010026265036176221,   0.011907295024053324,
        0.014122224106474948,  0.002881702206865988,   0.015412830209991379,
        0.010103981423178898,  0.014456791635032667,   0.005524526117380507,
        0.01705268049923486,   0.0041921026599170555,  0.016948576523323267,
        0.0006388486561331785, 0.010553286475190559,   0.0007133892106498745,
        0.0024233318590732977, 0.014850242737347993,   0.010990337068969767,
        0.017608218685683606,  0.006611126519572497,   0.01572371674353509,
        0.007633695625722481,  0.008784182331385141,   0.013172371756985176,
        0.009260310413545008,  0.009118474614461693,   0.003999255025992542,
        0.00805871916849993,   0.012060532966087953,   0.01093257082704249,
        0.011308052252542911,  0.011724261910912723,   0.015931347506601745,
        0.00481886983917667,   0.011137519648462487,   0.013433209253497114,
        0.0012171372798373784, 0.00020402085464400697, 0.015870477924163663,
        0.009256690167315111,  0.015759927458577648,   0.014287694295222688,
        0.016992607644848747,  0.0023881885544309945,  0.016808585390551835,
        0.010063572870545098,  0.014435235821481249,   0.008567994810313975,
        0.0052671965996169325, 0.015281060773776483,   0.006216762350864477,
        0.015781243511141633,  0.015503892390919654,   0.010393259446667769,
        0.002353621311222111,  0.0177087318336849,     0.012144367530965447,
        0.0022615160563713814, 0.017751977083877667,   0.004504770971105944,
        0.006437691509640454,  0.011857804520541328,   0.016556937238563635,
        0.005745936664732923,  0.011022659936585391,   0.0011965854829530036,
        0.014565042517693364,  0.014440621714829032,   0.010885847982597011,
        0.004183649107397657};

   public:
    mpi_op() : base_op() { c = mpi::rank; }

    void evaluate(machine::abstract_machine& rbm,
                  const machine::spin_state& state,
                  machine::rbm_context& context) override {
        auto& r = get_result_();
        r(0) = {data[c], 50 * ps[c]};
        c += mpi::n_proc;
    }

    double get_p() { return ps[c - mpi::n_proc]; }
};

void debug_mpi() {
    mpi::init(0, nullptr);
    model::kitaev k(2, -1);
    machine::rbm_base m(1, k.get_lattice());
    machine::spin_state s(m.n_visible);
    auto c = m.get_context(s);
    for (int j = 0; j < 1; j++) {
        std::cout.precision(10);
        mpi_op op;
        int pchs = 100 / mpi::n_proc;
        operators::aggregator agg(op, pchs);
        agg.track_variance();
        agg.set_zero();
        for (int i = 0; i < pchs; i++) {
            op.evaluate(m, s, c);
            agg.aggregate();
        }
        agg.finalize(100);
        std::cout.precision(17);
        mpi::cout << agg.get_result() << mpi::endl;
        mpi::cout << agg.get_variance(true) << mpi::endl;
    }
    mpi::end();
}

bool special_cases() {
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
        return true;
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
        return true;
    }

    if (ini::exact_energy && mpi::master && ini::model == ini::KITAEV) {
        std::unique_ptr<model::abstract_model> model;
        std::cout.precision(16);
        init_model(model);
        double e = dynamic_cast<model::kitaev*>(model.get())->exact_energy();
        std::cout << "Exact Energy: " << e << std::endl;
        return true;
    }

    if (ini::store_state) {
        store_state();
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    debug_mpi();
    return 0;
    mpi::init(argc, argv);
    int rc = ini::load(argc, argv);
    if (rc != 0) {
        return rc;
    }

    omp_set_num_threads(omp_get_max_threads());

    if (special_cases()) {
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
    std::unique_ptr<optimizer::base_plugin> plug;

    size_t seed = ini::seed;
    // Init Model
    rc |= init_model(model);

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
        std::unique_ptr<optimizer::base_plugin> best_plug;
        std::uniform_int_distribution<unsigned long> udist(0, ULONG_MAX);
        mpi::cout << "Seed Search: " << ini::seed_search_epochs << " Epochs"
                  << mpi::endl;
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
            rc |= init_optimizer(optimizer, model, rbm, sampler, plug);

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
                best_plug = std::move(plug);
                best_seed = seed;
            }

            rng.reset(0);
            rbm.reset(0);
            sampler.reset(0);
            optimizer.reset(0);
            plug.reset(0);
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
        plug = std::move(best_plug);
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
            rc |= init_optimizer(optimizer, model, rbm, sampler, plug);
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
    /* auto h = model->get_hamiltonian();
    sampler->clear_aggs();
    sampler->clear_ops();
    operators::derivative_op d{rbm->get_n_params()};
    operators::aggregator agg{d, sampler->get_my_n_samples()};
    operators::outer_aggregator_lazy agl{d, sampler->get_my_n_samples()};
    sampler->register_op(&d);
    sampler->register_agg(&agg);
    sampler->register_agg(&agl);
    agg.track_variance();
    sampler->sample();
    mpi::cout << agg.get_variance() << mpi::endl;
    std::cout << agg.get_result()(0) << std::endl;
    agl.finalize_diag(d.get_result());
    mpi::cout << agl.get_diag() << mpi::endl; */

    // auto& lat = model->get_lattice();
    // auto symm = lat.construct_symmetry({0.6});
    // auto symm_basis = lat.construct_symm_basis({0.6});

    // for (auto& sy : symm) {
    //     std::vector<size_t> x;
    //     for (auto i : symm_basis) x.push_back(sy.indices()(i));
    //     lat.print_lattice(x);
    // }

    // sampler->clear_aggs();
    // optimizer->optimize();

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

        time_keeper::clear();
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

        // mpi::cout <<
        // dynamic_cast<machine::rbm_base*>(rbm.get())->get_weights()
        //           << std::endl;
        //
        std::vector<operators::aggregator*> aggs;
        std::vector<operators::base_op*> ops;
        if (ini::model == ini::model_t::KITAEV) {
            auto hex = dynamic_cast<lattice::honeycomb*>(&model->get_lattice())
                           ->get_hexagons();
            for (auto& h : hex) {
                // for (auto& x : h) std::cout << x << ", ";
                // std::cout << std::endl;

                operators::local_op* op = new operators::local_op(h, plaq_op);
                operators::aggregator* agg =
                    new operators::aggregator(*op, sampler->get_my_n_samples());
                ops.push_back(op);
                aggs.push_back(agg);
            }

            sampler->register_ops(ops);
            sampler->register_aggs(aggs);
        }

        model->remove_helper_hamiltoian();
        auto& h = model->get_hamiltonian();
        operators::aggregator ah(h, sampler->get_my_n_samples());
        size_t b = 50;
        while (sampler->get_my_n_samples() % b != 0) b--;
        ah.track_variance(b);
        sampler->register_op(&h);
        sampler->register_agg(&ah);

        for (size_t i = 0; i < 1; i++) {
            sampler->sample();

            if (mpi::master) {
                std::cout.precision(16);
                if (ini::model == ini::model_t::KITAEV) {
                    for (size_t i = 0; i < aggs.size(); i++) {
                        std::cout << "Hex" << i << ": " << aggs[i]->get_result()
                                  << std::endl;
                    }
                }

                // std::cout << ah.get_result() / rbm->n_visible << std::endl;
                double std = ah.get_stddev()(0) / rbm->n_visible;
                double var = ah.get_variance()(0) / rbm->n_visible;
                double ene = std::real(ah.get_result()(0)) / rbm->n_visible;
                std::cout << "Var: " << var << std::endl;
                std::cout << "Std: " << std << std::endl;
                std::cout << "Tau: " << ah.get_tau() << std::endl;
                if (ini::sa_type == ini::sampler_t::METROPOLIS) {
                    std::cout << "Acc: "
                              << dynamic_cast<sampler::metropolis_sampler*>(
                                     sampler.get())
                                     ->get_acceptance_rate()
                              << std::endl;
                }
                int ndigits = std::log10(std);
                if (ndigits < 0) {
                    ndigits = -ndigits + 2;
                } else {
                    ndigits = 0;
                }
                std::cout << "E:   " << std::scientific
                          << std::setprecision(ndigits) << ene;
                std::cout << " ± " << std::setprecision(1) << std << std::endl;

                // std::cout << rbm->get_pfaffian().get_weights() << std::endl;
            }
        }
    }

    mpi::end();
    return rc;
    // size_t nch = 100;
    // auto samples = {1 << 9, 1 << 10, 1 << 11, 1 << 12};
    // for (auto& s : samples) {
    //     size_t k = std::log2l(s) - 2;
    //     std::vector<double> varxs(k);
    //     std::vector<double> taus(k);
    //     std::vector<double> varvarxs(k);
    //     std::vector<double> vartaus(k);
    //     for (size_t i = 0; i < k; i++) {
    //         varxs[i] = 0;
    //         taus[i] = 0;
    //         varvarxs[i] = 0;
    //         vartaus[i] = 0;
    //     }
    //     for (size_t ch = 0; ch < nch; ch++) {
    //         sampler->clear_ops();
    //         sampler->clear_aggs();

    //         sampler->set_n_samples(s);

    //         auto& h = model->get_hamiltonian();
    //         operators::aggregator ah(h);
    //         ah.track_variance();
    //         sampler->register_op(&h);
    //         sampler->register_agg(&ah);

    //         sampler->sample();

    //         std::cout << s << ", ";
    //         std::cout << std::real(ah.get_result()(0)) / rbm->n_visible << ",
    //         "; std::cout << ah.get_variance()(0) / rbm->n_visible <<
    //         std::endl;

    //         double E0 = std::abs(ah.get_result()(0));

    //         auto es = ah.get_resx();

    //         std::vector<std::complex<double>> locals(k);
    //         std::vector<double> vars(k);
    //         for (size_t j = 0; j < k; j++) {
    //             locals[j] = 0;
    //             vars[j] = 0;
    //         }
    //         for (size_t i = 0; i < es.size(); i++) {
    //             for (size_t j = 0; j < k; j++) {
    //                 locals[j] += es[i];
    //                 if (i % (1 << j) == (size_t)((1 << j) - 1)) {
    //                     vars[j] +=
    //                         std::pow(std::abs(locals[j] / (double)(1 << j)),
    //                         2);
    //                     locals[j] = 0;
    //                 }
    //             }
    //         }
    //         for (size_t j = 0; j < k; j++) {
    //             vars[j] /= ((double)s / (1 << j));
    //             vars[j] -= std::pow(E0, 2);
    //             double std = std::sqrt(vars[j] / ((double)s / (1 << j)));
    //             double tau = 0.5 * ((1 << j) * vars[j] / vars[0] - 1);
    //             varxs[j] += std;
    //             varvarxs[j] += std::pow(std, 2);
    //             taus[j] += tau;
    //             vartaus[j] += std::pow(tau, 2);
    //         }
    //     }
    //     for (size_t i = 0; i < k; i++) {
    //         varxs[i] /= nch;
    //         taus[i] /= nch;
    //         varvarxs[i] /= nch;
    //         vartaus[i] /= nch;
    //         varvarxs[i] -= std::pow(varxs[i], 2);
    //         vartaus[i] -= std::pow(taus[i], 2);
    //         std::cout << (1 << i) << ": \t" << varxs[i] << ", " <<
    //         varvarxs[i]
    //                   << " \t" << taus[i] << ", " << vartaus[i] << std::endl;
    //     }
    // }
}
