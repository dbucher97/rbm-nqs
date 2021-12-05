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

#include <algorithm>
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <istream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>
//
#include <tools/ini.hpp>

#include "boost/program_options/value_semantic.hpp"

namespace ini {

// Program variables

// Program
size_t seed = 421390484L;
size_t n_threads = 4;
std::string name = "";
std::string ini_file = "";
bool train = false;
bool evaluate = false;
bool store_state = false;
bool store_samples = false;
bool noprogress = false;
bool print_bonds = false;
bool print_hex = false;
bool deterministic = true;
bool exact_energy = false;
int seed_search = 0;
size_t seed_search_epochs = 1000;

// Model
model_t model = KITAEV;
size_t n_cells = 2;
int n_cells_b = -1;
coupling_t J = {{-1.}};
double h = 0.;
double helper_strength = 0.;
symmetry_t symmetry = {{0.5}};
std::string lattice_type = "";

// RBM
rbm_t rbm = BASIC;
size_t alpha = 1;
bool rbm_force = false;
size_t rbm_pop_mode = 0;
size_t rbm_cosh_mode = 2;
double rbm_weights = 1e-4;
double rbm_weights_imag = -1;
std::string rbm_weights_init_type = "";
size_t rbm_correlators = 0;
bool rbm_pfaffian = false;
symmetry_t rbm_pfaffian_symmetry = {{0.5}};
bool rbm_pfaffian_normalize = false;
double rbm_pfaffian_weights = 0.01;
bool rbm_pfaffian_no_updating = false;
std::string rbm_file_name = "";
std::string rbm_pfaffian_load = "";

// Sampler
sampler_t sa_type = METROPOLIS;
size_t sa_n_samples = 1000;
size_t sa_eval_samples = 0;
size_t sa_metropolis_samples_per_chain = 0;
size_t sa_metropolis_n_chains = 4;
size_t sa_metropolis_n_warmup_steps = 100;
size_t sa_metropolis_n_steps_per_sample = 1;
size_t sa_full_n_parallel_bits = 2;
std::string sa_exact_gs_file = "";
double sa_metropolis_bond_flips = 0.5;
int sa_pfaffian_refresh = 0;
int sa_lut_exchange = 0;

// Optimizer
optimizer_t opt_type = SR;
std::string opt_plugin = "";
decay_t opt_lr = {1e-2};
decay_t opt_sr_reg1 = {1, 1e-4, 0.9};
decay_t opt_sr_reg2 = {1e-3, 1e-6, 0.9};
decay_t opt_sr_deltareg1 = {1e-2};
double opt_sgd_real_factor = 1.;
double opt_adam_beta1 = 0.9;
double opt_adam_beta2 = 0.999;
double opt_adam_eps = 1e-8;
double opt_mom_alpha = 0.9;
double opt_mom_dialup = 1.;
std::string opt_sr_method = "cg";
bool opt_sr_iterative = true;
size_t opt_sr_max_iterations = 0;
double opt_sr_rtol = 0.;
double opt_heun_eps = 1e-3;
bool opt_resample = false;
double opt_resample_alpha1 = 0.1;
double opt_resample_alpha2 = 5;
double opt_resample_alpha3 = 10;

// Train
size_t n_epochs = 4000;

}  // namespace ini

void ini::parse_ini_file(int argc, char* argv[]) {
    namespace po = boost::program_options;
    // Try to load only ini file from command line options, ignore all other
    // options.
    po::options_description desc("Allowed options");
    desc.add_options()("infile,i", po::value(&ini_file), "ini file for params");
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .allow_unregistered()
                  .run(),
              vm);
    po::notify(vm);
    if (ini_file.length() != 0) {
        std::ifstream file{ini_file};
        if (!file.good()) {
            throw std::runtime_error("INI File '" + ini_file +
                                     "' does not exist!");
        }
        file.close();
    }
}

int ini::load(int argc, char* argv[]) {
    namespace po = boost::program_options;
    try {
        // Define all options
        parse_ini_file(argc, argv);
        po::options_description desc("Allowed options");
        // clang-format off
    desc.add_options()
    // Program
    ("help,h",                                "produce help message")
    ("train",                                 po::bool_switch(&train),                      "Train the RBM.")
    ("evaluate",                              po::bool_switch(&evaluate),                   "Evaluate RBM energy with statistics-")
    ("store_state",                           po::bool_switch(&store_state),                "Stores the RBM state into a file binary file 'name.state' and exit.")
    ("store_samples",                         po::bool_switch(&store_samples),              "Stores the samples into a plain text file 'name.samples' and exit.")
    ("print_bonds",                           po::bool_switch(&print_bonds),                "Print the bonds of the current model and exit.")
    ("print_hex"  ,                           po::bool_switch(&print_hex),                  "Print the hexagons and exit (only Kitaev).")
    ("exact_energy"  ,                        po::bool_switch(&exact_energy),               "Calculate the exact energy and exit (only Kitaev).")
    ("seed",                                  po::value(&seed),                             "Seed of the RNG.")
    ("deterministic",                         po::value(&deterministic),                    "Use specified seeds or seed RNG with random device.")
    ("infile,i",                              po::value<std::string>(),                     "'.ini' file with params.")
    ("name,n",                                po::value(&name),                             "Set name of the RBM.")
    ("n_threads,t",                           po::value(&n_threads),                        "Set number of OMP threads.")
    ("noprogress,P",                          po::bool_switch(&noprogress),                 "Turn off progress bar.")
    ("seed_search",                           po::value(&seed_search),                      "Set the number of different seeds to try.")
    ("seed_search_epochs",                    po::value(&seed_search_epochs),               "Set the number of epochs after which seeds are compared.")
    // Model
    ("model.type",                            po::value(&model),                            "Set the model type (kitaev or toric).")
    ("model.n_cells,c",                       po::value(&n_cells),                          "Set number of unit cells (square)")
    ("model.n_cells_b",                       po::value(&n_cells_b),                        "Set number of unit cells in the first dimension (if set to -1, use n_cells)")
    ("model.J",                               po::value(&J)->multitoken(),                  "Set Interaction strength.")
    ("model.h",                               po::value(&h),                                "Set second interaction strength.")
    ("model.helper_strength",                 po::value(&helper_strength),                  "Set P_W strength.")
    ("model.symmetry",                        po::value(&symmetry)->multitoken(),           "Set translational symmetry e.g. (0, 0.5, 1, 2).")
    ("model.lattice_type",                    po::value(&lattice_type),                     "Set lattice type if special type is available (honeycomb: hex)")
    // RBM
    ("rbm.type",                              po::value(&rbm),                              "Set RBM type (basic, symmetry, pfaffian, file).")
    ("rbm.alpha",                             po::value(&alpha),                            "Set number of hidden units (as multiple of visible units).")
    ("rbm.force,f",                           po::bool_switch(&rbm_force),                  "Force retraining of RBM")
    ("rbm.weights",                           po::value(&rbm_weights),                      "Set stddev for weights initialization")
    ("rbm.weights_imag",                      po::value(&rbm_weights_imag),                 "Set stddev for imag weights initialization (if not set = rbm.weights)")
    ("rbm.weights_type",                      po::value(&rbm_weights_init_type),            "Initialization type for RBM weights, for special initial states (deprecated).")
    ("rbm.correlators",                       po::value(&rbm_correlators),                  "Enables correlators if set to 1 and correlators are available for the model (deprecated).")
    ("rbm.pop_mode",                          po::value(&rbm_pop_mode),                     "Switches between Psi calculation modes (0 = sum log cosh, 1 = prod cosh).")
    ("rbm.cosh_mode",                         po::value(&rbm_cosh_mode),                    "Switches between Cosh modes (0 = std cosh + log, 1 = approx cosh, 2 = our log cosh)")
    ("rbm.pfaffian",                          po::value(&rbm_pfaffian),                     "Enables use of pfaffian wave function addition.")
    ("rbm.pfaffian.symmetry",                 po::value(&rbm_pfaffian_symmetry)->multitoken(),"Set symmetry for the pfaffian parameters.")
    ("rbm.pfaffian.weights",                  po::value(&rbm_pfaffian_weights),             "Set stddev of pfaffian parameters.")
    ("rbm.pfaffian.normalize",                po::value(&rbm_pfaffian_normalize),           "Normalize pfaffian parameters to pfaffian prop to 1 (deptrecated).")
    ("rbm.file.name",                         po::value(&rbm_file_name),                    "Specify the filename of the state to load from.")
    ("rbm.pfaffian.load",                     po::value(&rbm_pfaffian_load),                "Specify name of a already trained '.rbm' of a Pfaffian wavefunction to load.")
    ("rbm.pfaffian.no_updating",              po::bool_switch(&rbm_pfaffian_no_updating),   "Don't update the pfaffian context each iteration but calculate from scratch")
    // Sampler
    ("sampler.type",                          po::value(&sa_type),                          "Set sampler type (metropolis, full).")
    ("sampler.n_samples",                     po::value(&sa_n_samples),                     "Set number of samples.")
    ("sampler.n_samples_per_chain",           po::value(&sa_metropolis_samples_per_chain),  "Set number samples per chain (overrides sampler.n_samples).")
    ("sampler.eval_samples",                  po::value(&sa_eval_samples),                  "Set number of samples for evaluation.")
    ("sampler.full.n_parallel_bits",          po::value(&sa_full_n_parallel_bits),          "Set number of bits executed in parallel in perfect sampling. #MPI processes = 2^n.")
    ("sampler.metropolis.n_chains",           po::value(&sa_metropolis_n_chains),           "Set number of chains in Metropolis sampling.")
    ("sampler.metropolis.n_warmup_steps",     po::value(&sa_metropolis_n_warmup_steps),     "Set number of warmup sweeps.")
    ("sampler.metropolis.n_steps_per_sample", po::value(&sa_metropolis_n_steps_per_sample), "Set number of seeps between a sample.")
    // ("sampler.exact.gs_file",              po::value(&sa_exact_gs_file),                 "Set file of ground state for exact sampling")
    ("sampler.metropolis.bond_flips",         po::value(&sa_metropolis_bond_flips),         "Probability for bond flip for update proposal.")
    ("sampler.pfaffian_refresh",              po::value(&sa_pfaffian_refresh),              "Set number of Xinv updates before recalculation from scratch.")
    ("sampler.lut_exchange",                  po::value(&sa_lut_exchange),                  "Set number of samples before RBM LUT exchange is triggered.")
    // Optimizer
    ("optimizer.type",                        po::value(&opt_type),                         "Set optimizer type (SR, SGD).")
    ("optimizer.learning_rate,l",             po::value(&opt_lr)->multitoken(),             "Set learning rate, optionally with decay factor.")
    ("optimizer.sr.reg1",                     po::value(&opt_sr_reg1)->multitoken(),        "set regularization diagonal scaling decay rate, optionally with decay factor.")
    ("optimizer.sr.reg2",                     po::value(&opt_sr_reg2)->multitoken(),        "set regularization diagonal shift decay rate, optionally with decay factor.")
    ("optimizer.sr.deltareg1",                po::value(&opt_sr_deltareg1)->multitoken(),   "Diagonal scaling offset for pfaffian parameters.")
    ("optimizer.sr.method",                   po::value(&opt_sr_method),                    "The method for the SR solver (direct, minresqlp, cg, cg-direct).")
    ("optimizer.sr.iterative",                po::value(&opt_sr_iterative),                 "Backwards compatibility for non-MPI version (dummy parameter, uses default sr_method)")
    ("optimizer.sr.max_iterations",           po::value(&opt_sr_max_iterations),            "Set number of max iterations for iterative method.")
    ("optimizer.sr.rtol",                     po::value(&opt_sr_rtol),                      "Set residue tolerance for the iterative method.")
    ("optimizer.sgd.real_factor",             po::value(&opt_sgd_real_factor),              "Set the factor the real part of the update vector is divided by (default 1.).")
    ("optimizer.plugin",                      po::value(&opt_plugin),                       "Set optional plugin for optimization (momentum, adam, heun)")
    ("optimizer.adam.beta1",                  po::value(&opt_adam_beta1),                   "Set Adam beta1")
    ("optimizer.adam.beta2",                  po::value(&opt_adam_beta2),                   "Set Adam beta2")
    ("optimizer.adam.eps",                    po::value(&opt_adam_eps),                     "Set Adam eps")
    ("optimizer.mom.alpha",                   po::value(&opt_mom_alpha),                    "Set momentum alpha")
    ("optimizer.mom.dialup",                  po::value(&opt_mom_dialup),                   "Set momentum dialup")
    ("optimizer.heun.eps",                    po::value(&opt_heun_eps),                     "Set heun epsilon")
    ("optimizer.resample",                    po::value(&opt_resample),                     "Resample if certain conditions on energy / variance are not fullfilled (not recommended!).")
    ("optimizer.resample.alpha1",             po::value(&opt_resample_alpha1),              "Resample condition: energy difference smapller than alpha1")
    ("optimizer.resample.alpha2",             po::value(&opt_resample_alpha2),              "Resample condition: imaginary energy samller than alpha2 * variance")
    ("optimizer.resample.alpha3",             po::value(&opt_resample_alpha3),              "Resample condition: variance ratio samller than alpha3")
    // Train
    ("n_epochs,e",                            po::value(&n_epochs),                         "Set number of epochs for training.");
        // clang-format on
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        // Also parse ini file if available.
        if (ini_file.length() > 0) {
            std::ifstream config_stream(ini_file);
            po::store(po::parse_config_file(config_stream, desc), vm);
            config_stream.close();
        }

        po::notify(vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 1;
        }

        // Generate a name `n[n_cells]_h[rbm.n_hidden]` if not set.
        if (!vm.count("name")) {
            std::ostringstream oss;
            oss << "n" << n_cells << "_a" << alpha;
            name = oss.str();
        }

        if (sa_metropolis_samples_per_chain) {
            sa_n_samples =
                sa_metropolis_n_chains * sa_metropolis_samples_per_chain;
        }

    } catch (const po::unknown_option& e) {
        std::cerr << e.what();
        return -1;
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << std::endl;
        return -2;
    }

    return 0;
}

std::istream& ini::operator>>(std::istream& is, ini::rbm_t& rbm) {
    std::string token;
    is >> token;
    if (token == "basic") {
        rbm = ini::rbm_t::BASIC;
    } else if (token == "symmetry") {
        rbm = ini::rbm_t::SYMMETRY;
    } else if (token == "pfaffian") {
        rbm = ini::rbm_t::PFAFFIAN;
    } else if (token == "file") {
        rbm = ini::rbm_t::FILE;
    } else {
        throw std::runtime_error("RBM Type '" + token + "' not available!");
    }
    return is;
}

std::istream& ini::operator>>(std::istream& is, ini::sampler_t& sa) {
    std::string token;
    is >> token;
    if (token == "full") {
        sa = ini::sampler_t::FULL;
    } else if (token == "exact") {
        sa = ini::sampler_t::EXACT;
    } else if (token == "metropolis") {
        sa = ini::sampler_t::METROPOLIS;
    } else {
        throw std::runtime_error("Sampler Type '" + token + "' not available!");
    }
    return is;
}

std::istream& ini::operator>>(std::istream& is, ini::optimizer_t& opt) {
    std::string token;
    is >> token;
    if (token == "SR") {
        opt = ini::optimizer_t::SR;
    } else if (token == "SGD") {
        opt = ini::optimizer_t::SGD;
    } else {
        throw std::runtime_error("Optimizer Type '" + token +
                                 "' not available!");
    }
    return is;
}

std::istream& ini::operator>>(std::istream& is, ini::model_t& model) {
    std::string token;
    is >> token;
    if (token == "kitaev") {
        model = ini::model_t::KITAEV;
    } else if (token == "kitaevS3") {
        model = ini::model_t::KITAEV_S3;
    } else if (token == "isingS3") {
        model = ini::model_t::ISING_S3;
    } else if (token == "toric") {
        model = ini::model_t::TORIC;
    } else {
        throw std::runtime_error("Model Type '" + token + "' not available!");
    }
    return is;
}

void ini::validate(boost::any& v, const std::vector<std::string>& values,
                   ini::decay_t*, int) {
    std::vector<std::string> vals;

    // Split all strings by comma, for better support of ini loading.
    for (std::string s : values) {
        size_t pos = 0;
        std::string token;
        while ((pos = s.find(',')) != std::string::npos) {
            token = s.substr(0, pos);
            vals.push_back(token);
            s.erase(0, pos + 1);
        }
        vals.push_back(s);
    }
    ini::decay_t t;
    t.initial = std::stod(vals[0]);
    // if size of string is 3 load min and decay, otherwise set min and decay
    // to inital and 1. for a constant val.
    if (vals.size() == 3) {
        t.min = std::stod(vals[1]);
        t.decay = std::stod(vals[2]);
    } else {
        t.min = t.initial;
        t.decay = 1.;
    }
    v = t;
}

void ini::validate(boost::any& v, const std::vector<std::string>& values,
                   ini::coupling_t*, int) {
    std::vector<std::string> vals;

    // Split all strings by comma, for better support of ini loading.
    for (std::string s : values) {
        size_t pos = 0;
        std::string token;
        while ((pos = s.find(',')) != std::string::npos) {
            token = s.substr(0, pos);
            vals.push_back(token);
            s.erase(0, pos + 1);
        }
        vals.push_back(s);
    }
    ini::coupling_t x;

    std::transform(vals.begin(), vals.end(), std::back_inserter(x.strengths),
                   [](const std::string& x) { return std::stod(x); });

    v = x;
}

void ini::validate(boost::any& v, const std::vector<std::string>& values,
                   ini::symmetry_t*, int) {
    std::vector<std::string> vals;

    // Split all strings by comma, for better support of ini loading.
    for (std::string s : values) {
        size_t pos = 0;
        std::string token;
        while ((pos = s.find(',')) != std::string::npos) {
            token = s.substr(0, pos);
            vals.push_back(token);
            s.erase(0, pos + 1);
        }
        vals.push_back(s);
    }

    std::vector<double> vec(vals.size());

    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = std::stod(vals[i]);
    }

    v = symmetry_t{vec};
}
