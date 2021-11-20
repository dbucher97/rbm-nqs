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

#ifndef MAX_SPIN_SITES
#define MAX_SPIN_SITES 128
#endif

#pragma once

#include <Eigen/Dense>
#include <bitset>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace machine {

struct spin_state {
    size_t n;
    std::bitset<MAX_SPIN_SITES> bitset;

    spin_state(size_t n, size_t v = 0);
    spin_state(const spin_state& other);
    spin_state& operator=(const spin_state& other);

    bool operator==(const spin_state& other) const;
    bool operator[](size_t i) const;
    std::bitset<MAX_SPIN_SITES>::reference operator[](size_t i);

    Eigen::VectorXcd to_vec() const;

    size_t to_num() const;

    std::vector<size_t> to_indices() const;

    void flip(size_t i);
    void flip(const std::vector<size_t>& i);

    size_t to_num_loc(const std::vector<size_t>& acts) const;

    static void flips_from_loc(size_t loc, std::vector<size_t>& flips,
                               const std::vector<size_t>& acts);
    void set_random(std::mt19937& rng);

    std::string to_string() const;
};
}  // namespace machine

namespace std {
template <>
struct hash<machine::spin_state> {
    auto operator()(const machine::spin_state& k) const {
        return std::hash<std::bitset<MAX_SPIN_SITES>>()(k.bitset);
    }
};
}  // namespace std
