#include "manifest.h"

#include <algorithm>
#include <fcntl.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unistd.h>

namespace amind {

Manifest::Manifest(const std::filesystem::path& dataDir) : dataDir_(dataDir) {}

bool Manifest::load() {
    std::filesystem::path manifestPath = dataDir_ / FILENAME;
    if (!std::filesystem::exists(manifestPath)) {
        return false;
    }

    std::ifstream file(manifestPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open manifest file: " + manifestPath.string());
    }

    nlohmann::json j = nlohmann::json::parse(file);
    file.close();

    checkpointSeq_ = j.value("checkpoint_seq", uint64_t{0});
    nextSstId_ = j.value("next_sst_id", uint64_t{0});
    l0Files_ = j.value("l0", std::vector<std::string>{});
    l1Files_ = j.value("l1", std::vector<std::string>{});
    l2Files_ = j.value("l2", std::vector<std::string>{});

    return true;
}

void Manifest::save() {
    std::filesystem::path tempPath = dataDir_ / TEMP_FILENAME;
    std::filesystem::path finalPath = dataDir_ / FILENAME;

    nlohmann::json j;
    j["version"] = CURRENT_VERSION;
    j["checkpoint_seq"] = checkpointSeq_;
    j["next_sst_id"] = nextSstId_;
    j["l0"] = l0Files_;
    j["l1"] = l1Files_;
    j["l2"] = l2Files_;

    std::ofstream tempFile(tempPath, std::ios::binary);
    if (!tempFile.is_open()) {
        throw std::runtime_error("Failed to create temporary manifest file: " + tempPath.string());
    }
    tempFile << j.dump(2) << '\n';
    tempFile.flush();
    tempFile.close();

    int fd = open(tempPath.c_str(), O_RDONLY);
    if (fd != -1) {
        fsync(fd);
        close(fd);
    }

    std::filesystem::rename(tempPath, finalPath);

    int dirFd = open(dataDir_.c_str(), O_RDONLY);
    if (dirFd != -1) {
        fsync(dirFd);
        close(dirFd);
    }
}

void Manifest::addL0File(const std::string& filename) {
    l0Files_.insert(l0Files_.begin(), filename);
}

void Manifest::compactL0ToL1(const std::vector<std::string>& oldL0Files,
                            const std::string& newL1File) {
    for (const auto& oldFile : oldL0Files) {
        auto it = std::find(l0Files_.begin(), l0Files_.end(), oldFile);
        if (it != l0Files_.end()) {
            l0Files_.erase(it);
        }
    }
    l1Files_.insert(l1Files_.begin(), newL1File);
}

void Manifest::compactL1ToL2(const std::vector<std::string>& oldL1Files,
                            const std::string& newL2File) {
    for (const auto& oldFile : oldL1Files) {
        auto it = std::find(l1Files_.begin(), l1Files_.end(), oldFile);
        if (it != l1Files_.end()) {
            l1Files_.erase(it);
        }
    }
    l2Files_.insert(l2Files_.begin(), newL2File);
}

uint64_t Manifest::allocateSstId() {
    return nextSstId_++;
}

}  // namespace amind