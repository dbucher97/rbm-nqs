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
#pragma once

#include <mpi.h>

#include <iostream>

namespace mpi {
extern int rank;
extern int n_proc;
extern bool master;

extern void init(int argc, char* argv[]);
extern void end();

struct ostream {};
enum streamflag { ENDL, FLUSH };
extern mpi::ostream cout;
extern mpi::streamflag flush, endl;

inline ostream& operator<<(ostream& os, const streamflag& t) {
    if (mpi::master) switch (t) {
            case ENDL:
                std::cout << std::endl;
            case FLUSH:
                std::cout << std::flush;
        }
    return os;
}
template <typename T>
inline ostream& operator<<(ostream& os, const T& t) {
    if (mpi::master) std::cout << t;
    return os;
}

}  // namespace mpi
