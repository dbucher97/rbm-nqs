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
#include <tools/mpi.hpp>

/**
 * @brief A logger singleton class. `init` needs to be called to open the
 * output stream.
 */
class logger {
    std::ofstream out;          ///< output file stream.
    std::ostringstream buffer;  ///< stringstream line buffer.
    std::ostringstream header;  ///< stringstream header buffer.
    bool first_line = true;     ///< is the first line, flag for header.

   public:
    /**
     * @brief Returns the logger instance
     *
     * @return Reference to the logger instance.
     */
    static logger& get() {
        static logger _instance;
        return _instance;
    }

    /**
     * @brief Destructor of the reference instance.
     */
    ~logger();

    /**
     * @brief init instance method.
     */
    void init_();
    /**
     * @brief write a line to the log.
     *
     * @param std::string The line.
     */
    void writeln(const std::string&);
    /**
     * @brief newline instance method.
     */
    void newline_();

    /**
     * @brief log message instance method.
     *
     * @tparam T type of message.
     * @param t message to log.
     */
    template <typename T>
    void log_(const T& t) {
        buffer << t << ", ";
    }

    /**
     * @brief log message with header instance method.
     *
     * @tparam T type of message.
     * @param t message to log.
     * @param name header name.
     */
    template <typename T>
    void log_(const T& t, const std::string& name) {
        // if first line, log to header buffer
        if (first_line) {
            header << name << ", ";
        }
        buffer << t << ", ";
    }

    /**
     * @brief log message static method.
     *
     * @tparam T type of message.
     * @param t message to log.
     */
    template <typename T>
    static void log(const T& t) {
        if (mpi::master) {
            get().log_(t);
        }
    }

    /**
     * @brief log message with header static method.
     *
     * @tparam T type of message.
     * @param t message to log.
     * @param name header name.
     */
    template <typename T>
    static void log(const T& t, const std::string& name) {
        if (mpi::master) {
            get().log_(t, name);
        }
    }

    /**
     * @brief newline static method
     */
    static void newline() {
        if (mpi::master) {
            get().newline_();
        }
    }
    /**
     * @brief init static method
     */
    static void init() {
        if (mpi::master) get().init_();
    }

   private:
    // Private constructors and assign operator to allow only for singleton.
    logger() {}
    logger(const logger&);
    logger& operator=(const logger&);
};

/**
 * @brief `<<` wrapper for logger.
 *
 * @tparam T typename of message to log.
 * @param l logger instance reference.
 * @param t message to log.
 *
 * @return logger instance reference.
 */
template <typename T>
logger& operator<<(logger& l, const T& t) {
    if (mpi::master) {
        l.log_(t);
    }
    return l;
}
