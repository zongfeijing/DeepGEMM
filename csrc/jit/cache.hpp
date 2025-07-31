#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include "kernel_runtime.hpp"

namespace deep_gemm {

// Custom hasher for std::filesystem::path that works regardless of standard library support
struct filesystem_path_hash {
    std::size_t operator()(const std::filesystem::path& p) const noexcept {
        return std::filesystem::hash_value(p);
    }
};

class KernelRuntimeCache {
    std::unordered_map<std::filesystem::path, std::shared_ptr<KernelRuntime>, filesystem_path_hash> cache;

public:
    // TODO: consider cache capacity
    KernelRuntimeCache() = default;

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        // Hit the runtime cache
        if (const auto& iterator = cache.find(dir_path); iterator != cache.end())
            return iterator->second;

        if (KernelRuntime::check_validity(dir_path))
            return cache[dir_path] = std::make_shared<KernelRuntime>(dir_path);
        return nullptr;
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

} // namespace deep_gemm
