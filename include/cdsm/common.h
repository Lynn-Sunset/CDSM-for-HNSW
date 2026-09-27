#pragma once
#include <faiss/IndexHNSW.h>
#include <faiss/impl/DistanceComputer.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

inline void need(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
