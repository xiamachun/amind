#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace amind {

namespace detail {

inline const uint32_t* getCRC32Table() {
    static uint32_t table[256];
    static std::once_flag initFlag;
    std::call_once(initFlag, []() {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t crc = i;
            for (int j = 0; j < 8; j++) {
                if (crc & 1) {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
            table[i] = crc;
        }
    });
    return table;
}

} // namespace detail

/**
 * @brief 一次性计算数据的 CRC32 校验值
 * @param data 数据指针
 * @param len 数据长度
 * @return CRC32 校验值
 */
inline uint32_t crc32(const void* data, size_t len) {
    const uint8_t* buf = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    const uint32_t* table = detail::getCRC32Table();

    for (size_t i = 0; i < len; i++) {
        crc = table[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
    }

    return crc ^ 0xFFFFFFFF;
}

/**
 * @brief 流式 CRC32 计算器，支持分段更新
 */
class CRC32Builder {
public:
    CRC32Builder() : crc_(0xFFFFFFFF) {}

    /**
     * @brief 更新 CRC32 状态
     * @param data 数据指针
     * @param len 数据长度
     */
    void update(const void* data, size_t len) {
        const uint8_t* buf = static_cast<const uint8_t*>(data);
        const uint32_t* table = detail::getCRC32Table();

        for (size_t i = 0; i < len; i++) {
            crc_ = table[(crc_ ^ buf[i]) & 0xFF] ^ (crc_ >> 8);
        }
    }

    /**
     * @brief 完成计算并返回最终 CRC32 值
     * @return CRC32 校验值
     */
    uint32_t finalize() const {
        return crc_ ^ 0xFFFFFFFF;
    }

private:
    uint32_t crc_;
};

} // namespace amind
