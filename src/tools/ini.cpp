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

#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <vector>
//
#include <tools/ini.hpp>

namespace ini {

// Porgram
size_t seed = 421390484L;
size_t n_threads = 8;
std::string name = "";
std::string ini_file = "";
bool train = false;

// Model
size_t n_cells = 2;
double J = -1.;

// RBM
rbm_t rbm = SYMMETRY;
size_t n_hidden = 3;
bool rbm_force = false;
double rbm_weights = 0.0001;
double rbm_weights_imag = -1;

// Sampler
sampler_t sa_type = METROPOLIS;
size_t sa_n_samples = 1000;
size_t sa_metropolis_n_chains = 8;
size_t sa_metropolis_n_warmup_steps = 100;
size_t sa_metropolis_n_steps_per_sample = 5;
size_t sa_full_n_parallel_bits = 3;
std::string sa_exact_gs_file = "";

// Optimizer
std::string sr_plugin = "";
decay_t sr_lr = {0.001, 0.001, 1.};
decay_t sr_reg = {1, 1e-4, 0.9};

// Train
size_t n_epochs = 600;

}  // namespace ini

void ini::parse_ini_file(int argc, char* argv[]) {
    namespace po = boost::program_options;
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
        parse_ini_file(argc, argv);
        po::options_description desc("Allowed options");
        // clang-format off
    desc.add_options()
    // Program
    ("help,h", "produce help message")
    ("train", po::bool_switch(&train), "train the RBM")
    ("seed", po::value(&seed), "seed of the rng")
    ("infile,i", po::value<std::string>(), "ini file for params")
    ("name,n", po::value(&name), "set name of current rbm")
    ("n_threads,t", po::value(&n_threads), "set number of omp threads")
    // Model
    ("model.n_cells,c", po::value(&n_cells), "set number of unit cells in one dimension")
    ("model.J", po::value(&J), "Interaction coefficient")
    // RBM
    ("rbm.type", po::value(&rbm), "set rbm type")
    ("rbm.n_hidden", po::value(&n_hidden), "set number of hidden units")
    ("rbm.force,f", po::bool_switch(&rbm_force), "force retraining of RBM")
    ("rbm.weights", po::value(&rbm_weights), "set stddev for weights initialization")
    ("rbm.weights_imag", po::value(&rbm_weights_imag), "set stddev for imag weights initialization (if not set = rbm.weights)")
    // Sampler
    ("sampler.type", po::value(&sa_type), "set sampler type")
    ("sampler.n_samples", po::value(&sa_n_samples), "set sampler n sampler (metropolis only)")
    ("sampler.full.n_parallel_bits", po::value(&sa_full_n_parallel_bits), "set number of bits executed in parallel in full sampling")
    ("sampler.metropolis.n_chains", po::value(&sa_metropolis_n_chains), "set number of MCMC chains in Metropolis sampling")
    ("sampler.metropolis.n_warmup_steps", po::value(&sa_metropolis_n_warmup_steps), "set number of MCMC warmup steps")
    ("sampler.metropolis.n_steps_per_sample", po::value(&sa_metropolis_n_steps_per_sample), "set number of MCMC steps between a sample")
    ("sampler.exact.gs_file", po::value(&sa_exact_gs_file), "set file of ground state for exact sampling")
    // Optimizer
    ("stochastic_reconfiguration.learning_rate,l", po::value(&sr_lr)->multitoken(), "set learning rate optionally with decay factor")
    ("stochastic_reconfiguration.regularization,r", po::value(&sr_reg)->multitoken(), "set regularization diagonal shift decay rate optionally with decay factor")
    ("stochastic_reconfiguration.plugin", po::value(&sr_plugin), "set optional plugin for SR adam/momentum")
    // Train
    ("n_epochs,e", po::value(&n_epochs), "set number of epochs for training");
        // clang-format on
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

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

        if (!vm.count("name")) {
            std::ostringstream oss;
            oss << "n" << n_cells << "_h" << n_hidden;
            name = oss.str();
        }

    } catch (const po::unknown_option& e) {
        std::cerr << e.what();
        return -1;
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << std::endl;
        return -2;
    }

    std::cout << "Starting '" << name << "'!" << std::endl;

    return 0;
}

std::istream& ini::operator>>(std::istream& is, ini::rbm_t& rbm) {
    std::string token;
    is >> token;
    if (token == "basic") {
        rbm = ini::rbm_t::BASIC;
    } else if (token == "symmetry") {
        rbm = ini::rbm_t::SYMMETRY;
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

void ini::validate(boost::any& v, const std::vector<std::string>& values,
                   ini::decay_t*, int) {
    std::vector<std::string> vals;

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
    if (vals.size() == 3) {
        t.min = std::stod(vals[1]);
        t.decay = std::stod(vals[2]);
    } else {
        t.min = t.initial;
        t.decay = 1.;
    }
    v = t;
}
