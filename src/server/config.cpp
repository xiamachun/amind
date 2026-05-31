#include "config.h"

#include <cstdlib>
#include <fstream>
#include <spdlog/spdlog.h>
#include <sstream>

namespace amind {

Result<AppConfig> AppConfig::load(const std::string& path) {
    AppConfig config;
    std::ifstream file(path);
    if (!file.is_open()) {
        return makeError(Error::ConfigError, "cannot open config: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);

        // Trim whitespace
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t") + 1);
        };
        trim(key);
        trim(value);

        config.values_[key] = value;
    }

    // Environment variable overrides: AMIND_* prefix
    for (auto& [key, val] : config.values_) {
        std::string env_key = "AMIND_";
        for (char c : key) {
            env_key += (c == '.' || c == '-') ? '_' : static_cast<char>(toupper(c));
        }
        const char* env_val = std::getenv(env_key.c_str());
        if (env_val) {
            val = env_val;
            spdlog::info("Config override from env: {}={}", key, val);
        }
    }

    spdlog::info("Loaded config from {} ({} entries)", path, config.values_.size());
    return config;
}

std::string AppConfig::get(const std::string& key, const std::string& default_val) const {
    auto it = values_.find(key);
    return (it != values_.end()) ? it->second : default_val;
}

int AppConfig::getInt(const std::string& key, int default_val) const {
    auto val = get(key);
    if (val.empty()) return default_val;
    try { return std::stoi(val); } catch (...) { return default_val; }
}

float AppConfig::getFloat(const std::string& key, float default_val) const {
    auto val = get(key);
    if (val.empty()) return default_val;
    try { return std::stof(val); } catch (...) { return default_val; }
}

}  // namespace amind
