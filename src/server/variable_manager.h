#pragma once

#include "core/result.h"

#include <functional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace amind {

class AppConfig;

enum class VarType  { STRING, INT, FLOAT, BOOL };
enum class VarMode  { DYNAMIC, READONLY };

struct Variable {
    std::string name;
    std::string value;
    std::string default_value;
    VarType     type{VarType::STRING};
    VarMode     mode{VarMode::READONLY};
    std::string description;
    std::string category;
};

/// Central variable registry with SHOW / SET / RELOAD support.
/// Thread-safe (shared_mutex: many readers, exclusive writer).
class VariableManager {
public:
    VariableManager() = default;

    /// Register a variable definition (called during Engine::init).
    void registerVar(const std::string& name, VarType type, VarMode mode,
                     const std::string& default_val,
                     const std::string& description,
                     const std::string& category);

    /// Batch-load current values from an AppConfig.
    void loadFrom(const AppConfig& config);

    // ── Read ────────────────────────────────────────────────────────────

    /// SHOW VARIABLES LIKE 'pattern' (% = wildcard, empty = all).
    std::vector<Variable> list(const std::string& pattern = "%") const;

    std::string get(const std::string& name) const;
    int         getInt(const std::string& name) const;
    float       getFloat(const std::string& name) const;

    /// Check if a variable exists.
    bool has(const std::string& name) const;

    // ── Write ───────────────────────────────────────────────────────────

    /// SET a variable. Returns error if READONLY or name unknown.
    Result<void, Error> set(const std::string& name, const std::string& value);

    /// Persist all current values back to amind.conf (preserving comments).
    Result<void, Error> persistToFile(const std::string& config_path) const;

    /// Reload DYNAMIC variables from a config file.
    Result<int, Error> reloadFromFile(const std::string& config_path);

    // ── Change notification ─────────────────────────────────────────────

    using ChangeCallback = std::function<void(const std::string& name,
                                              const std::string& new_value)>;
    void onChange(const std::string& name, ChangeCallback cb);

private:
    bool matchPattern(const std::string& name, const std::string& pattern) const;

    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, Variable> vars_;
    std::unordered_map<std::string, std::vector<ChangeCallback>> callbacks_;
};

}  // namespace amind
