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

//
#include <tools/ini.hpp>
#include <tools/logger.hpp>

logger::~logger() {
    // Close file on destruct.
    if (out.is_open()) {
        out.flush();
        out.close();
    }
}

void logger::init_() {
    std::string filename = ini::name + ".log";
    std::ifstream file{filename};
    std::cout << "Logging to '" << filename << "'!" << std::endl;

    // Append to file if log already exists and no force is demanded.
    if (file.good() && !ini::rbm_force)
        out = std::ofstream(filename, std::ios::app);
    else
        out = std::ofstream(filename);
    buffer.precision(16);
}

void logger::writeln(const std::string& str) {
    // Write to output stream if open otherwise log to cout.
    if (out.is_open()) {
        out << str << std::endl;
    } else {
        std::cout << str << std::endl;
    }
}

void logger::newline_() {
    // Remove the last 2 to characters from the line (', ').
    // Write the header line first if `first_line`, also add '#'.
    std::string s;
    if (first_line) {
        s = header.str();
        if (s.length() > 2) {
            s.pop_back();
            s.pop_back();
            writeln("# " + s);
        }
        first_line = false;
    }
    s = buffer.str();
    if (s.length() > 2) {
        s.pop_back();
        s.pop_back();
    }
    // Write string and reset buffer.
    writeln(s);
    buffer.str(std::string());
}
