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
#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
//

class logger {
    std::ofstream out;
    std::ostringstream buffer;
    std::ostringstream header;
    bool first_line = true;

   public:
    static logger& get() {
        static logger _instance;
        return _instance;
    }
    ~logger();

    void init_();
    void writeln(const std::string&);
    void newline_();

    template <typename T>
    void log_(const T& t) {
        buffer << t << ", ";
    }

    template <typename T>
    void log_(const T& t, const std::string& name) {
        if (first_line) {
            header << name << ", ";
        }
        buffer << t << ", ";
    }

    template <typename T>
    static void log(const T& t) {
        get().log_(t);
    }

    template <typename T>
    static void log(const T& t, const std::string& name) {
        get().log_(t, name);
    }

    static void newline() { get().newline_(); }
    static void init() { get().init_(); }

   private:
    logger() {}
    logger(const logger&);
    logger& operator=(const logger&);
};

template <typename T>
logger& operator<<(logger& l, const T& t) {
    l.log_(t);
    return l;
}
