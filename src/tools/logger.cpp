/**
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
    if (out.is_open()) {
        out.flush();
        out.close();
    }
}

void logger::init_() {
    std::string filename = ini::name + ".log";
    std::ifstream file{filename};
    std::cout << "Logging to '" << filename << "'!" << std::endl;
    if (file.good() && !ini::rbm_force)
        out = std::ofstream(filename, std::ios::app);
    else
        out = std::ofstream(filename);
}

void logger::writeln(const std::string& str) {
    if (out.is_open()) {
        out << str << std::endl;
    } else {
        std::cout << str << std::endl;
    }
}

void logger::newline_() {
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
    writeln(s);
    buffer.str(std::string());
}
