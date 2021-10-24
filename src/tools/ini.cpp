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
int seed_search = 0;
size_t seed_search_epochs = 200;

// Model
model_t model = KITAEV;
size_t n_cells = 2;
int n_cells_b = -1;
coupling_t J = {{-1.}};
double helper_strength = 0.;
symmetry_t symmetry = {{0.5}};
std::string lattice_type = "";

// RBM
rbm_t rbm = BASIC;
size_t alpha = 1;
bool rbm_force = false;
size_t rbm_pop_mode = 0;
size_t rbm_cosh_mode = 0;
double rbm_weights = 0.0001;
double rbm_weights_imag = -1;
std::string rbm_weights_init_type = "";
size_t rbm_correlators = 0;
bool rbm_pfaffian = false;
symmetry_t rbm_pfaffian_symmetry = {};
bool rbm_pfaffian_normalize = false;
double rbm_pfaffian_weights = 0.1;
bool rbm_pfaffian_no_updating = false;
std::string rbm_file_name = "";
std::string rbm_pfaffian_load = "";

// Sampler
sampler_t sa_type = METROPOLIS;
size_t sa_n_samples = 1000;
size_t sa_eval_samples = 0;
size_t sa_metropolis_samples_per_chain = 0;
size_t sa_metropolis_n_chains = 16;
size_t sa_metropolis_n_warmup_steps = 100;
size_t sa_metropolis_n_steps_per_sample = 10;
size_t sa_full_n_parallel_bits = 3;
std::string sa_exact_gs_file = "";
double sa_metropolis_bond_flips = 0.5;
int sa_pfaffian_refresh = 0;
int sa_lut_exchange = 0;

// Optimizer
optimizer_t opt_type = SR;
std::string opt_plugin = "";
decay_t opt_lr = {1e-2};
decay_t opt_sr_reg1 = {1e-4};
decay_t opt_sr_reg2 = {1e-3, 1e-6, 0.9};
decay_t opt_sr_deltareg1 = {1e-2};
double opt_sgd_real_factor = 1.;
double opt_adam_beta1 = 0.9;
double opt_adam_beta2 = 0.999;
double opt_adam_eps = 1e-8;
double opt_mom_alpha = 0.3;
std::string opt_sr_method = "minresqlp";
bool opt_sr_iterative = false;
size_t opt_sr_max_iterations = 0;
double opt_sr_rtol = 0.;
double opt_heun_eps = 1e-3;
bool opt_resample = false;
double opt_resample_alpha1 = 0.1;
double opt_resample_alpha2 = 5;
double opt_resample_alpha3 = 10;

// Train
size_t n_epochs = 600;

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
    ("train",                                 po::bool_switch(&train),                      "train the RBM")
    ("evaluate",                              po::bool_switch(&evaluate),                   "evaluate results of the RBM")
    ("store_state",                           po::bool_switch(&store_state),                "stores the state into a file 'name.state'")
    ("store_samples",                         po::bool_switch(&store_samples),              "stores the samples into a plain text file 'name.samples'")
    ("print_bonds",                           po::bool_switch(&print_bonds),                "print the bonds of the current model and exit")
    ("print_hex"  ,                           po::bool_switch(&print_hex),                  "print the hexagons of the kitaev model and exit")
    ("seed",                                  po::value(&seed),                             "seed of the rng")
    ("infile,i",                              po::value<std::string>(),                     "ini file for params")
    ("name,n",                                po::value(&name),                             "set name of current rbm")
    ("n_threads,t",                           po::value(&n_threads),                        "set number of omp threads")
    ("noprogress,P",                          po::bool_switch(&noprogress),                 "switch to turn off progress bar")
    ("seed_search",                           po::value(&seed_search),                      "set the number of different seeds to try")
    ("seed_search_epochs",                    po::value(&seed_search_epochs),               "set the number of epochs after which seeds are compared")
    // Model
    ("model.type",                            po::value(&model),                            "Model type.")
    ("model.n_cells,c",                       po::value(&n_cells),                          "set number of unit cells in one dimension")
    ("model.n_cells_b",                       po::value(&n_cells_b),                        "set number of unit cells in another dimension (if set to -1, use n_cells)")
    ("model.J",                               po::value(&J)->multitoken(),                                "Interaction strength")
    ("model.helper_strength",                 po::value(&helper_strength),                  "Helper Hamiltonian strength")
    ("model.symmetry",                        po::value(&symmetry)->multitoken(),           "Set translational symmetry")
    ("model.lattice_type",                    po::value(&lattice_type),                     "Set lattice type if special type is available (honeycomb -> hex base)")
    // RBM
    ("rbm.type",                              po::value(&rbm),                              "set rbm type")
    ("rbm.alpha",                             po::value(&alpha),                            "set number of hidden units = alpha * n_visbile")
    ("rbm.force,f",                           po::bool_switch(&rbm_force),                  "force retraining of RBM")
    ("rbm.weights",                           po::value(&rbm_weights),                      "set stddev for weights initialization")
    ("rbm.weights_imag",                      po::value(&rbm_weights_imag),                 "set stddev for imag weights initialization (if not set = rbm.weights)")
    ("rbm.weights_type",                      po::value(&rbm_weights_init_type),            "Initialization type for RBM weights, for special initial states")
    ("rbm.correlators",                       po::value(&rbm_correlators),                  "enables correlators if set to 1 and correlators are available for the model")
    ("rbm.pop_mode",                          po::value(&rbm_pop_mode),                     "switches between Psi calculation modes.")
    ("rbm.cosh_mode",                         po::value(&rbm_cosh_mode),                    "turns cosh approximation on")
    ("rbm.pfaffian",                          po::value(&rbm_pfaffian),                     "enables use of pfaffian wave function addition")
    ("rbm.pfaffian.symmetry",                 po::value(&rbm_pfaffian_symmetry)->multitoken(),"symmetry condition of pfaffian parameters")
    ("rbm.pfaffian.weights",                  po::value(&rbm_pfaffian_weights),             "stdev of pfaffian parameters")
    ("rbm.pfaffian.normalize",                po::value(&rbm_pfaffian_normalize),           "normalize pfaffian parameters to pfaffian prop to 1")
    ("rbm.file.name",                         po::value(&rbm_file_name),                    "specify the filename of the quantum state")
    ("rbm.pfaffian.load",                     po::value(&rbm_pfaffian_load),                "specify name of a already trained rbm file of type pfaffian to load into this")
    ("rbm.pfaffian.no_updating",              po::bool_switch(&rbm_pfaffian_no_updating),   "dont update the pfaffian context each iteration but calculate from scratch")
    // Sampler
    ("sampler.type",                          po::value(&sa_type),                          "set sampler type")
    ("sampler.n_samples",                     po::value(&sa_n_samples),                     "set sampler n sampler (metropolis only)")
    ("sampler.n_samples_per_chain",           po::value(&sa_metropolis_samples_per_chain),  "set n samples per chain (overrides sampler.n_samples)")
    ("sampler.eval_samples",                  po::value(&sa_eval_samples),                  "set sampler n sampler (metropolis only) for evaluation")
    ("sampler.full.n_parallel_bits",          po::value(&sa_full_n_parallel_bits),          "set number of bits executed in parallel in full sampling")
    ("sampler.metropolis.n_chains",           po::value(&sa_metropolis_n_chains),           "set number of MCMC chains in Metropolis sampling")
    ("sampler.metropolis.n_warmup_steps",     po::value(&sa_metropolis_n_warmup_steps),     "set number of MCMC warmup steps")
    ("sampler.metropolis.n_steps_per_sample", po::value(&sa_metropolis_n_steps_per_sample), "set number of MCMC steps between a sample")
    ("sampler.exact.gs_file",                 po::value(&sa_exact_gs_file),                 "set file of ground state for exact sampling")
    ("sampler.metropolis.bond_flips",         po::value(&sa_metropolis_bond_flips),         "probability for bond flips for update proposal")
    ("sampler.pfaffian_refresh",              po::value(&sa_pfaffian_refresh),              "set number of Xinv updates before recalculating from scratch")
    ("sampler.lut_exchange",              po::value(&sa_lut_exchange),                  "set number of samples before RBM LUT exchange is triggered")
    // Optimizer
    ("optimizer.type",                        po::value(&opt_type),                         "set optimizer type")
    ("optimizer.learning_rate,l",             po::value(&opt_lr)->multitoken(),             "set learning rate optionally with decay factor")
    ("optimizer.sr.reg1",                     po::value(&opt_sr_reg1)->multitoken(),         "set regularization diagonal scaling decay rate optionally with decay factor")
    ("optimizer.sr.reg2",                     po::value(&opt_sr_reg2)->multitoken(),         "set regularization diagonal shift decay rate optionally with decay factor")
    ("optimizer.sr.deltareg1",                po::value(&opt_sr_deltareg1)->multitoken(),   "diagonal scaling offset for pfaffian parameters")
    ("optimizer.sr.method",                   po::value(&opt_sr_method),                    "the method for the sr solver either: direct, minresqlp, cg")
    ("optimizer.sr.iterative",                po::value(&opt_sr_iterative),                 "backwards compatibility for non mpi version (dummy parameter, uses default sr_method)")
    ("optimizer.sr.max_iterations",           po::value(&opt_sr_max_iterations),            "set number of max iterations for iterative method")
    ("optimizer.sr.rtol",                     po::value(&opt_sr_rtol),                      "set residue tolerance for the iterative method")
    ("optimizer.sgd.real_factor",             po::value(&opt_sgd_real_factor),              "set the factor the real part of the update vector is divided by (default 1.)")
    ("optimizer.plugin",                      po::value(&opt_plugin),                       "set optional plugin for SR adam/momentum")
    ("optimizer.adam.beta1",                  po::value(&opt_adam_beta1),                   "set ADAM plug beta1")
    ("optimizer.adam.beta2",                  po::value(&opt_adam_beta2),                   "set ADAM plug beta2")
    ("optimizer.adam.eps",                    po::value(&opt_adam_eps),                     "set ADAM plug eps")
    ("optimizer.mom.alpha",                   po::value(&opt_mom_alpha),                    "set momentum plug alpha")
    ("optimizer.heun.eps",                    po::value(&opt_heun_eps),                     "set heun plug epsilon")
    ("optimizer.resample",                    po::value(&opt_resample),                     "resample if certain conditions on energy / variance are not fullfilled")
    ("optimizer.resample.alpha1",             po::value(&opt_resample_alpha1),              "resample condition: energy difference smapller than alpha1")
    ("optimizer.resample.alpha2",             po::value(&opt_resample_alpha2),              "resample condition: imaginary energy samller than alpha2 * variance")
    ("optimizer.resample.alpha3",             po::value(&opt_resample_alpha3),              "resample condition: variance ratio samller than alpha3")
    // Train
    ("n_epochs,e",                            po::value(&n_epochs),                         "set number of epochs for training");
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
