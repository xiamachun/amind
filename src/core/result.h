#pragma once

#include <cassert>
#include <optional>
#include <string>
#include <variant>

namespace amind {

/// Error type for all amind operations.
/// Uses error codes + optional message for fast, no-exception error handling.
struct Error {
    enum Code : uint16_t {
        Ok = 0,
        InvalidArgument,
        NotFound,
        AlreadyExists,
        CorruptedData,
        CrcMismatch,
        IOError,
        WalError,
        LsmError,
        VectorError,
        GraphError,
        ProviderError,
        ProviderTimeout,
        ProviderUnavailable,
        QueueFull,
        QueueEmpty,
        ConflictDetected,
        StaleMemory,
        SessionNotFound,
        SessionClosed,
        ConfigError,
        InternalError,
    };

    Code code{Ok};
    std::string message;

    Error() = default;
    Error(Code c, std::string msg = "") : code(c), message(std::move(msg)) {}

    [[nodiscard]] bool ok() const { return code == Ok; }
    [[nodiscard]] explicit operator bool() const { return !ok(); }

    [[nodiscard]] std::string toString() const {
        static const char* names[] = {
            "Ok", "InvalidArgument", "NotFound", "AlreadyExists",
            "CorruptedData", "CrcMismatch", "IOError", "WalError",
            "LsmError", "VectorError", "GraphError", "ProviderError",
            "ProviderTimeout", "ProviderUnavailable", "QueueFull", "QueueEmpty",
            "ConflictDetected", "StaleMemory", "SessionNotFound", "SessionClosed",
            "ConfigError", "InternalError",
        };
        auto idx = static_cast<size_t>(code);
        std::string result = (idx < sizeof(names) / sizeof(names[0]))
                                 ? names[idx]
                                 : "Unknown(" + std::to_string(idx) + ")";
        if (!message.empty()) {
            result += ": " + message;
        }
        return result;
    }
};

/// Result<T, E> — Rust-inspired result type for error handling without exceptions.
/// All fallible operations in amind return Result<T> (where E defaults to Error).
///
/// Usage:
///   Result<int> r = 42;                    // success
///   Result<int> r = makeError(Error::NotFound, "no such memory");
///   if (r.ok()) { use(*r); }
template <typename T, typename E = Error>
class Result {
public:
    // Success constructor (pass by value + move for both copy and move semantics)
    Result(T value) : data_(std::move(value)) {}  // NOLINT(implicit)

    // Error constructors
    Result(E error) : data_(std::move(error)) {}  // NOLINT(implicit)

    [[nodiscard]] bool ok() const { return std::holds_alternative<T>(data_); }
    [[nodiscard]] explicit operator bool() const { return ok(); }

    /// Access the value (UB if !ok())
    [[nodiscard]] T& value() & { assert(ok()); return std::get<T>(data_); }
    [[nodiscard]] const T& value() const& { assert(ok()); return std::get<T>(data_); }
    [[nodiscard]] T&& value() && { assert(ok()); return std::get<T>(std::move(data_)); }

    /// Access the error (UB if ok())
    [[nodiscard]] E& error() & { assert(!ok()); return std::get<E>(data_); }
    [[nodiscard]] const E& error() const& { assert(!ok()); return std::get<E>(data_); }

    /// Dereference operators (shorthand for value())
    [[nodiscard]] T& operator*() & { return value(); }
    [[nodiscard]] const T& operator*() const& { return value(); }
    [[nodiscard]] T&& operator*() && { return std::move(*this).value(); }
    [[nodiscard]] T* operator->() { return &value(); }
    [[nodiscard]] const T* operator->() const { return &value(); }

    /// Map: transform value if ok, pass error through
    template <typename F>
    auto map(F&& func) -> Result<decltype(func(std::declval<T>())), E> {
        if (ok()) return func(value());
        return error();
    }

private:
    std::variant<T, E> data_;
};

/// Specialization for void result (just success/failure, no value).
template <typename E>
class Result<void, E> {
public:
    Result() : err_(std::nullopt) {}
    Result(E error) : err_(std::move(error)) {}  // NOLINT(implicit)

    [[nodiscard]] bool ok() const { return !err_.has_value(); }
    [[nodiscard]] explicit operator bool() const { return ok(); }
    [[nodiscard]] E& error() { assert(!ok()); return *err_; }
    [[nodiscard]] const E& error() const { assert(!ok()); return *err_; }

private:
    std::optional<E> err_;
};

/// Helper to create error results concisely.
inline Error makeError(Error::Code code, std::string msg = "") {
    return Error{code, std::move(msg)};
}

/// Convenience: check and early-return on error.
/// Usage: AMIND_TRY(result);  — returns the error if result is not ok
#define AMIND_TRY(expr)                    \
    do {                                   \
        auto&& _r = (expr);               \
        if (!_r.ok()) return _r.error();   \
    } while (false)

/// AMIND_ASSIGN_OR_RETURN: assign value or return error.
/// Usage: AMIND_ASSIGN_OR_RETURN(auto val, some_fallible_call());
#define AMIND_ASSIGN_OR_RETURN(lhs, expr)  \
    auto _tmp_##__LINE__ = (expr);         \
    if (!_tmp_##__LINE__.ok())             \
        return _tmp_##__LINE__.error();    \
    lhs = std::move(*_tmp_##__LINE__)

}  // namespace amind
