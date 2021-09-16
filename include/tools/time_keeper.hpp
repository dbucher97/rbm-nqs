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
 */
#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace time_keeper {
using clock = std::chrono::steady_clock;
extern std::map<std::string, std::chrono::time_point<clock>> start_times;
extern std::map<std::string, int> counters;
extern std::map<std::string, double> elapsed_times;
extern std::vector<std::string> tracked;
extern int iteration_count;

extern void start(const std::string& name);
extern void end(const std::string& name);
extern void itn();
extern void clear();
extern void resumee();

}  // namespace time_keeper
