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

#include <machine/spin_state.hpp>
#include <random>

using namespace machine;

spin_state::spin_state(size_t n, size_t v) : n{n}, bitset(v) {}
spin_state::spin_state(const spin_state& other)
    : n{other.n}, bitset(other.bitset) {}
spin_state& spin_state::operator=(const spin_state& other) {
    n = other.n;
    bitset = other.bitset;
    return *this;
}
bool spin_state::operator==(const spin_state& other) const {
    return bitset == other.bitset;
}

bool spin_state::operator[](size_t i) const { return bitset[i]; }
std::bitset<MAX_SPIN_SITES>::reference spin_state::operator[](size_t i) {
    return bitset[i];
}

Eigen::VectorXcd spin_state::to_vec() const {
    Eigen::VectorXcd ret(n);
    for (size_t i = 0; i < n; i++) {
        ret(i) = (bitset[i] ? -1. : 1.);
    }
    return ret;
}

size_t spin_state::to_num() const { return bitset.to_ullong(); }

std::vector<size_t> spin_state::to_indices() const {
    std::vector<size_t> ret;
    for (size_t i = 0; i < n; i++) {
        if (bitset[i]) ret.push_back(i);
    }

    return ret;
}

void spin_state::flip(size_t i) { bitset[i] = !bitset[i]; }
void spin_state::flip(const std::vector<size_t>& flips) {
    for (auto& i : flips) bitset[i] = !bitset[i];
}

size_t spin_state::to_num_loc(const std::vector<size_t>& acts_on) const {
    size_t res = 0;
    for (size_t i = 0; i < acts_on.size(); i++) {
        if (bitset[acts_on[i]]) res |= (1 << i);
    }
    return res;
}

void spin_state::flips_from_loc(size_t loc, std::vector<size_t>& flips,
                                const std::vector<size_t>& acts_on) {
    flips.clear();
    for (size_t i = 0; i < acts_on.size(); i++) {
        if ((loc >> i) & 1) flips.push_back(acts_on[i]);
    }
}

void spin_state::set_random(std::mt19937& rng) {
    std::uniform_int_distribution<> dist(0, 1);
    for (size_t i = 0; i < n; i++) {
        bitset |= dist(rng) << i;
    }
}

std::string spin_state::to_string() const {
    return bitset.to_string().substr(MAX_SPIN_SITES - n);
}
