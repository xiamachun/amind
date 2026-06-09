#include "variable_manager.h"
#include "config.h"
#include "core/file_utils.h"

#include <algorithm>
#include <fstream>
#include <spdlog/spdlog.h>
#include <sstream>

namespace amind {

void VariableManager::registerVar(const std::string& name, VarType type,
                                   VarMode mode, const std::string& default_val,
                                   const std::string& description,
                                   const std::string& category) {
    std::unique_lock lock(mutex_);
    Variable v;
    v.name = name;
    v.value = default_val;
    v.default_value = default_val;
    v.type = type;
    v.mode = mode;
    v.description = description;
    v.category = category;
    vars_[name] = std::move(v);
}

void VariableManager::loadFrom(const AppConfig& config) {
    std::unique_lock lock(mutex_);
    for (auto& [name, var] : vars_) {
        auto val = config.get(name, var.default_value);
        if (!val.empty()) var.value = val;
    }
}

std::vector<Variable> VariableManager::list(const std::string& pattern) const {
    std::shared_lock lock(mutex_);
    std::vector<Variable> result;
    for (const auto& [name, var] : vars_) {
        if (pattern == "%" || pattern.empty() || matchPattern(name, pattern)) {
            result.push_back(var);
        }
    }
    std::sort(result.begin(), result.end(),
              [](const Variable& a, const Variable& b) {
                  if (a.category != b.category) return a.category < b.category;
                  return a.name < b.name;
              });
    return result;
}

std::string VariableManager::get(const std::string& name) const {
    std::shared_lock lock(mutex_);
    auto it = vars_.find(name);
    return (it != vars_.end()) ? it->second.value : "";
}

int VariableManager::getInt(const std::string& name) const {
    auto val = get(name);
    try { return std::stoi(val); } catch (...) { return 0; }
}

float VariableManager::getFloat(const std::string& name) const {
    auto val = get(name);
    try { return std::stof(val); } catch (...) { return 0.0f; }
}

bool VariableManager::has(const std::string& name) const {
    std::shared_lock lock(mutex_);
    return vars_.count(name) > 0;
}

Result<void, Error> VariableManager::set(const std::string& name,
                                          const std::string& value) {
    std::vector<ChangeCallback> callbacks_to_fire;
    {
        std::unique_lock lock(mutex_);
        auto it = vars_.find(name);
        if (it == vars_.end()) {
            return makeError(Error::ConfigError,
                             "Unknown variable: " + name);
        }
        if (it->second.mode == VarMode::READONLY) {
            return makeError(Error::ConfigError,
                             "Variable '" + name + "' is READONLY (requires restart to change)");
        }

        if (it->second.type == VarType::INT) {
            try { std::stoi(value); } catch (...) {
                return makeError(Error::ConfigError,
                                 "Invalid integer value for '" + name + "': " + value);
            }
        } else if (it->second.type == VarType::FLOAT) {
            try { std::stof(value); } catch (...) {
                return makeError(Error::ConfigError,
                                 "Invalid float value for '" + name + "': " + value);
            }
        }

        std::string old_value = it->second.value;
        it->second.value = value;
        spdlog::info("SET {} = {} (was {})", name, value, old_value);

        auto cb_it = callbacks_.find(name);
        if (cb_it != callbacks_.end()) {
            callbacks_to_fire = cb_it->second;
        }
    }
    // Fire callbacks outside lock to avoid blocking readers
    for (auto& cb : callbacks_to_fire) {
        try { cb(name, value); } catch (const std::exception& e) {
            spdlog::error("Variable callback error for '{}': {}", name, e.what());
        }
    }
    return Result<void, Error>();
}

Result<void, Error> VariableManager::persistToFile(const std::string& config_path) const {
    // Read original file, update values in-place (preserve comments/structure)
    std::ifstream in(config_path);
    if (!in.is_open()) {
        return makeError(Error::IOError, "cannot open config for writing: " + config_path);
    }

    std::shared_lock lock(mutex_);

    std::ostringstream out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '#') {
            auto eq = line.find('=');
            if (eq != std::string::npos) {
                std::string key = line.substr(0, eq);
                // trim
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);

                auto it = vars_.find(key);
                if (it != vars_.end()) {
                    out << key << " = " << it->second.value << "\n";
                    continue;
                }
            }
        }
        out << line << "\n";
    }
    in.close();
    lock.unlock();

    std::string tmp_path = config_path + ".tmp";
    std::ofstream of(tmp_path, std::ios::trunc);
    if (!of.is_open()) {
        return makeError(Error::IOError, "cannot write config tmp: " + tmp_path);
    }
    of << out.str();
    of.close();
    // fsync + atomic rename
    amind::fsyncFile(tmp_path);
    std::rename(tmp_path.c_str(), config_path.c_str());
    spdlog::info("Config persisted to {} (atomic)", config_path);
    return Result<void, Error>();
}

Result<int, Error> VariableManager::reloadFromFile(const std::string& config_path) {
    // Parse fresh config
    auto config_result = AppConfig::load(config_path);
    if (!config_result.ok()) return config_result.error();

    int changed = 0;
    std::unique_lock lock(mutex_);
    for (auto& [name, var] : vars_) {
        if (var.mode != VarMode::DYNAMIC) continue;

        auto new_val = config_result->get(name, var.default_value);
        if (new_val != var.value) {
            std::string old_val = var.value;
            var.value = new_val;
            changed++;

            auto cb_it = callbacks_.find(name);
            if (cb_it != callbacks_.end()) {
                for (auto& cb : cb_it->second) {
                    try { cb(name, new_val); } catch (...) {}
                }
            }
            spdlog::info("RELOAD: {} = {} (was {})", name, new_val, old_val);
        }
    }
    return changed;
}

void VariableManager::onChange(const std::string& name, ChangeCallback cb) {
    std::unique_lock lock(mutex_);
    callbacks_[name].push_back(std::move(cb));
}

bool VariableManager::matchPattern(const std::string& name,
                                    const std::string& pattern) const {
    // Simple SQL LIKE: % = any chars, _ = single char
    // For simplicity, support only prefix%, %suffix, %contains%
    if (pattern == "%" || pattern.empty()) return true;

    std::string p = pattern;
    bool prefix_wild = (!p.empty() && p.front() == '%');
    bool suffix_wild = (!p.empty() && p.back() == '%');

    if (prefix_wild) p = p.substr(1);
    if (suffix_wild && !p.empty()) p = p.substr(0, p.size() - 1);

    if (p.empty()) return true;

    if (prefix_wild && suffix_wild) {
        return name.find(p) != std::string::npos;
    } else if (prefix_wild) {
        return name.size() >= p.size() &&
               name.compare(name.size() - p.size(), p.size(), p) == 0;
    } else if (suffix_wild) {
        return name.compare(0, p.size(), p) == 0;
    } else {
        return name == p;
    }
}

}  // namespace amind
