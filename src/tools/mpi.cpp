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
 **/

#include <iostream>
//
#include <tools/mpi.hpp>

namespace mpi {
int rank = -1;
int n_proc = -1;
bool master = false;
mpi::ostream cout;
mpi::streamflag flush = FLUSH;
mpi::streamflag endl = ENDL;

}  // namespace mpi

using namespace mpi;

void mpi::init(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi::rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi::n_proc);
    master = (rank == 0);
}
void mpi::end() {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
