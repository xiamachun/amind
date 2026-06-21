#pragma once

#include "core/result.h"

#include <string>
#include <unordered_map>

namespace amind {

/// amind version string (single source of truth).
constexpr const char* AMIND_VERSION = "0.6.0";

/// Application configuration loaded from amind.conf + environment overrides.
class AppConfig {
public:
    /// Load config from file path.
    static Result<AppConfig> load(const std::string& path);

    std::string get(const std::string& key, const std::string& default_val = "") const;
    int getInt(const std::string& key, int default_val = 0) const;
    float getFloat(const std::string& key, float default_val = 0.0f) const;

private:
    std::unordered_map<std::string, std::string> values_;
};

}  // namespace amind
