#pragma once

#include <exception>
#include <string>
#include <sstream>

class KUException final : public std::exception {
    std::string message = {};

public:
    template<typename... Args>
    explicit KUException(const char *name, const char* file, const int line, Args&&... args) {
        std::ostringstream oss;
        
        oss << name << " error (" << file << ":" << line << "): ";
        (oss << ... << args);
        message = oss.str();
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#define THROW_KU_EXCEPTION(name, ...) \
    throw KUException(name, __FILE__, __LINE__, __VA_ARGS__)

// This `KU_ASSERT` is triggered no matter if the code is compiled with `-DNDEBUG` or not.
#define KU_ASSERT(cond, ...)                                                                                      \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion `%s` failed (%s:%d): ", #cond, __FILE__, __LINE__);          \
            if constexpr (sizeof(#__VA_ARGS__) > 1) {                                                \
                fprintf(stderr, ", " __VA_ARGS__);                                                        \
            }                                                                                             \
            fprintf(stderr, "\n");                                                                       \
            THROW_KU_EXCEPTION("Assertion", "Assertion `", #cond, "` failed.");                          \
        }                                                                                                 \
    } while(0)
