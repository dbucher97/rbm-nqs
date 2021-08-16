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

#include <omp.h>

#include <chrono>
#include <cstdio>
#include <tools/mpi.hpp>
#include <tools/time_keeper.hpp>

namespace time_keeper {
std::map<std::string, std::chrono::time_point<clock>> start_times = {};
std::map<std::string, int> counters = {};
std::map<std::string, double> elapsed_times = {};
std::vector<std::string> tracked = {};
int iteration_count = 0;
}  // namespace time_keeper

void time_keeper::start(const std::string& name) {
    if (omp_get_thread_num() == 0) start_times[name] = clock::now();
}
void time_keeper::end(const std::string& name) {
    if (omp_get_thread_num() == 0) {
        if (elapsed_times.find(name) == elapsed_times.end()) {
            elapsed_times[name] = 0.;
            counters[name] = 0;
            tracked.push_back(name);
        }
        elapsed_times[name] +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                clock::now() - start_times[name])
                .count() /
            1000000.;
        counters[name]++;
    }
}

void time_keeper::itn() { iteration_count++; }

void time_keeper::resumee() {
    std::printf("\n%-20s | %-10s | %-10s | %-10s\n", "NAME", "T / E", "T / C",
                "C / E");
    std::printf(
        "-------------------- | ---------- | ---------- | ----------\n");
    for (const auto& name : tracked) {
        double time_per_epoch = elapsed_times[name] / iteration_count;
        double calls_per_epoch = ((double)counters[name]) / iteration_count;
        double time_per_call = elapsed_times[name] / counters[name];
        std::printf("%-20s | %10.2f | %10.5f | %10.1f\n", name.c_str(),
                    time_per_epoch, time_per_call, calls_per_epoch);
    }
}
