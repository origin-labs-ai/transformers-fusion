#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/production.h"
#include <iostream>
#include <cassert>
#include <string>
#include <stdexcept>
#include <sstream>

namespace quant {
    // Helper to capture logger output if possible, assuming logger writes to std::cerr or std::cout, or just test its functionality.
    // If not, we just test the API.
}

int main() {
    std::cout << "[Test] Running Debug verification test..." << std::endl;
    
    quant::Logger& logger = quant::Logger::instance();
    logger.set_level(quant::Logger::DEBUG);
    logger.log(quant::Logger::INFO, "Test info log");
    logger.log(quant::Logger::DEBUG, "Test debug log");
    
    // Config test
    quant::AppConfig cfg;
    cfg.set("test_key", "42");
    assert(cfg.get_int("test_key") == 42);
    
    cfg.set("float_key", "3.14");
    assert(std::abs(cfg.get_float("float_key") - 3.14f) < 1e-4f);
    
    cfg.set("str_key", "value");
    assert(cfg.get_string("str_key") == "value");

    // Exception handling test
    bool caught = false;
    try {
        throw std::runtime_error("Test Exception");
    } catch (const std::runtime_error& e) {
        caught = true;
        assert(std::string(e.what()) == "Test Exception");
    }
    assert(caught);

    std::cout << "Debug test passed!" << std::endl;
    return 0;
}
