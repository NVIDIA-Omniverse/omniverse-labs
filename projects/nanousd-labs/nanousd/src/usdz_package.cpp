// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/usdz_package.h"
#include "nanousd/resource.h"
#include "nanousd/usda_writer.h"
#include "nanousd/usdc_writer.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace nanousd {

namespace {

constexpr uint32_t kEocdSignature = 0x06054b50u;
constexpr uint32_t kCentralSignature = 0x02014b50u;
constexpr uint32_t kLocalSignature = 0x04034b50u;
constexpr uint16_t kCompressionStored = 0;
constexpr uint16_t kEncryptedFlag = 0x0001u;
constexpr uint64_t kUsdzAlignment = 64;
constexpr uint16_t kZipVersion20 = 20;
constexpr uint16_t kPaddingExtraFieldId = 0xffffu;

struct ZipEntry {
    std::string name;
    uint16_t flags = 0;
    uint16_t method = 0;
    uint32_t compressedSize = 0;
    uint32_t uncompressedSize = 0;
    uint32_t localHeaderOffset = 0;
};

struct ZipDirectory {
    std::string packagePath;
    std::string filePath;
    std::shared_ptr<std::vector<uint8_t>> data;
    std::vector<ZipEntry> entries;
};

struct ZipCentralWriteEntry {
    std::string name;
    uint32_t crc = 0;
    uint32_t size = 0;
    uint32_t localHeaderOffset = 0;
    uint16_t localExtraSize = 0;
};

uint16_t ReadU16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) |
           static_cast<uint16_t>(p[1] << 8);
}

uint32_t ReadU32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

void AppendU16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xffu));
}

void AppendU32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xffu));
}

uint32_t Crc32(const std::vector<uint8_t>& data) {
    uint32_t crc = 0xffffffffu;
    for (uint8_t b : data) {
        crc ^= b;
        for (int i = 0; i < 8; ++i) {
            uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

bool EndsWithSlash(const std::string& s) {
    return !s.empty() && (s.back() == '/' || s.back() == '\\');
}

std::string LowerAscii(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

bool HasLowerExtension(std::string_view path, std::string_view ext) {
    std::string lower = LowerAscii(path);
    return lower.size() >= ext.size() &&
           lower.compare(lower.size() - ext.size(), ext.size(), ext) == 0;
}

uint16_t ComputeLocalExtraSize(uint64_t localHeaderOffset,
                               uint16_t nameSize,
                               uint32_t dataSize) {
    uint64_t endWithoutExtra = localHeaderOffset + 30u + nameSize + dataSize;
    uint16_t extraSize = static_cast<uint16_t>(
        (kUsdzAlignment - (endWithoutExtra % kUsdzAlignment)) %
        kUsdzAlignment);
    if (extraSize > 0 && extraSize < 4) extraSize += static_cast<uint16_t>(kUsdzAlignment);
    return extraSize;
}

bool AppendPaddingExtraField(std::vector<uint8_t>& out, uint16_t extraSize) {
    if (extraSize == 0) return true;
    if (extraSize < 4) return false;

    AppendU16(out, kPaddingExtraFieldId);
    AppendU16(out, static_cast<uint16_t>(extraSize - 4));
    out.insert(out.end(), static_cast<size_t>(extraSize - 4), uint8_t{0});
    return true;
}

bool ValidateAndNormalizeEntries(const std::vector<UsdzFileEntry>& entries,
                                 std::vector<UsdzFileEntry>* normalized,
                                 std::string* error) {
    if (entries.empty()) {
        if (error) *error = "USDZ package must contain a root layer entry";
        return false;
    }
    if (entries.size() > 0xffffu) {
        if (error) *error = "USDZ package exceeds 32-bit ZIP entry count";
        return false;
    }

    normalized->clear();
    normalized->reserve(entries.size());
    for (const auto& entry : entries) {
        std::string name = NormalizePackageEntryPath(entry.path);
        if (name.empty() || EndsWithSlash(entry.path)) {
            if (error) *error = "USDZ package entries must be files with non-empty paths";
            return false;
        }
        if (name.size() > 0xffffu) {
            if (error) *error = "USDZ entry path is too long: " + name;
            return false;
        }
        if (entry.data.size() >
                static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            if (error) *error = "USDZ entry is too large for 32-bit ZIP: " + name;
            return false;
        }
        for (const auto& existing : *normalized) {
            if (existing.path == name) {
                if (error) *error = "Duplicate USDZ entry path: " + name;
                return false;
            }
        }
        normalized->push_back({std::move(name), entry.data});
    }
    return true;
}

bool AppendLocalFileHeader(std::vector<uint8_t>& out,
                           const UsdzFileEntry& entry,
                           ZipCentralWriteEntry* central,
                           std::string* error) {
    if ((out.size() % kUsdzAlignment) != 0) {
        if (error) *error = "Internal USDZ writer alignment error";
        return false;
    }
    if (out.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        if (error) *error = "USDZ package exceeds 32-bit ZIP offset range";
        return false;
    }

    central->name = entry.path;
    central->crc = Crc32(entry.data);
    central->size = static_cast<uint32_t>(entry.data.size());
    central->localHeaderOffset = static_cast<uint32_t>(out.size());
    central->localExtraSize =
        ComputeLocalExtraSize(central->localHeaderOffset,
                              static_cast<uint16_t>(central->name.size()),
                              central->size);

    AppendU32(out, kLocalSignature);
    AppendU16(out, kZipVersion20);
    AppendU16(out, 0); // flags
    AppendU16(out, kCompressionStored);
    AppendU16(out, 0); // modification time
    AppendU16(out, 0); // modification date
    AppendU32(out, central->crc);
    AppendU32(out, central->size);
    AppendU32(out, central->size);
    AppendU16(out, static_cast<uint16_t>(central->name.size()));
    AppendU16(out, central->localExtraSize);
    out.insert(out.end(), central->name.begin(), central->name.end());
    if (!AppendPaddingExtraField(out, central->localExtraSize)) {
        if (error) *error = "Failed to write USDZ local padding field";
        return false;
    }
    out.insert(out.end(), entry.data.begin(), entry.data.end());
    return true;
}

bool AppendCentralDirectory(std::vector<uint8_t>& out,
                            const std::vector<ZipCentralWriteEntry>& central,
                            std::string* error) {
    if (out.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        if (error) *error = "USDZ package exceeds 32-bit ZIP offset range";
        return false;
    }
    uint32_t centralOffset = static_cast<uint32_t>(out.size());

    for (const auto& entry : central) {
        AppendU32(out, kCentralSignature);
        AppendU16(out, kZipVersion20); // version made by
        AppendU16(out, kZipVersion20); // version needed
        AppendU16(out, 0); // flags
        AppendU16(out, kCompressionStored);
        AppendU16(out, 0); // modification time
        AppendU16(out, 0); // modification date
        AppendU32(out, entry.crc);
        AppendU32(out, entry.size);
        AppendU32(out, entry.size);
        AppendU16(out, static_cast<uint16_t>(entry.name.size()));
        AppendU16(out, 0); // central extra length
        AppendU16(out, 0); // comment length
        AppendU16(out, 0); // disk start
        AppendU16(out, 0); // internal attributes
        AppendU32(out, 0); // external attributes
        AppendU32(out, entry.localHeaderOffset);
        out.insert(out.end(), entry.name.begin(), entry.name.end());
    }

    if (out.size() - centralOffset >
            static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        if (error) *error = "USDZ central directory exceeds 32-bit ZIP size range";
        return false;
    }
    uint32_t centralSize = static_cast<uint32_t>(out.size() - centralOffset);

    AppendU32(out, kEocdSignature);
    AppendU16(out, 0); // disk number
    AppendU16(out, 0); // central directory disk number
    AppendU16(out, static_cast<uint16_t>(central.size()));
    AppendU16(out, static_cast<uint16_t>(central.size()));
    AppendU32(out, centralSize);
    AppendU32(out, centralOffset);
    AppendU16(out, 0); // no comment; EOCD ends the file
    return true;
}

std::vector<uint8_t> WriteRootLayerBytes(const Layer& rootLayer,
                                         std::string_view rootEntryPath,
                                         std::string* error) {
    if (HasLowerExtension(rootEntryPath, ".usda")) {
        std::string text = WriteUsda(rootLayer);
        if (text.empty()) {
            if (error) *error = "Failed to serialize USDZ root layer as USDA";
            return {};
        }
        return std::vector<uint8_t>(text.begin(), text.end());
    }
    if (HasLowerExtension(rootEntryPath, ".usdc") ||
        HasLowerExtension(rootEntryPath, ".usd")) {
        std::vector<uint8_t> data = WriteUsdc(rootLayer);
        if (data.empty()) {
            if (error) *error = "Failed to serialize USDZ root layer as USDC";
            return {};
        }
        return data;
    }

    if (error) {
        *error = "USDZ root layer entry must use .usd, .usda, or .usdc: ";
        error->append(rootEntryPath);
    }
    return {};
}

std::optional<ZipDirectory> ParseCentralDirectory(const uint8_t* central,
                                                  size_t centralSize,
                                                  uint16_t totalEntries,
                                                  const std::string& packagePath,
                                                  std::string* error) {
    ZipDirectory dir;
    dir.packagePath = packagePath;
    dir.entries.reserve(totalEntries);

    size_t pos = 0;
    for (uint16_t i = 0; i < totalEntries; ++i) {
        if (pos + 46 > centralSize ||
            ReadU32(central + pos) != kCentralSignature) {
            if (error) *error = "Malformed ZIP central directory in: " + packagePath;
            return std::nullopt;
        }

        const uint8_t* h = central + pos;
        ZipEntry entry;
        entry.flags = ReadU16(h + 8);
        entry.method = ReadU16(h + 10);
        entry.compressedSize = ReadU32(h + 20);
        entry.uncompressedSize = ReadU32(h + 24);
        const uint16_t nameLen = ReadU16(h + 28);
        const uint16_t extraLen = ReadU16(h + 30);
        const uint16_t commentLen = ReadU16(h + 32);
        const uint16_t diskStart = ReadU16(h + 34);
        entry.localHeaderOffset = ReadU32(h + 42);

        pos += 46;
        if (pos + nameLen + extraLen + commentLen > centralSize) {
            if (error) *error = "Malformed ZIP central directory entry in: " + packagePath;
            return std::nullopt;
        }

        entry.name.assign(reinterpret_cast<const char*>(central + pos), nameLen);
        entry.name = NormalizePackageEntryPath(entry.name);
        pos += nameLen + extraLen + commentLen;

        if (entry.compressedSize == 0xffffffffu ||
            entry.uncompressedSize == 0xffffffffu ||
            entry.localHeaderOffset == 0xffffffffu) {
            if (error) *error = "ZIP64 entries are not supported: " + entry.name;
            return std::nullopt;
        }
        if (diskStart != 0) {
            if (error) *error = "Multi-disk ZIP entries are not supported: " + entry.name;
            return std::nullopt;
        }
        if (entry.flags & kEncryptedFlag) {
            if (error) *error = "Encrypted USDZ entries are not supported: " + entry.name;
            return std::nullopt;
        }
        if (entry.method != kCompressionStored) {
            if (error) *error = "Compressed USDZ entries are not supported: " + entry.name;
            return std::nullopt;
        }

        dir.entries.push_back(std::move(entry));
    }

    return dir;
}

std::optional<ZipDirectory> ReadCentralDirectoryFromBuffer(const uint8_t* data,
                                                           size_t size,
                                                           const std::string& packagePath,
                                                           std::string* error) {
    if (size < 22) {
        if (error) *error = "File is too small to be a ZIP package: " + packagePath;
        return std::nullopt;
    }

    const uint64_t fileSize = static_cast<uint64_t>(size);
    const uint64_t tailSize = std::min<uint64_t>(fileSize, 22u + 0xffffu);
    const uint8_t* tail = data + (fileSize - tailSize);

    size_t eocdPos = std::numeric_limits<size_t>::max();
    for (size_t i = static_cast<size_t>(tailSize) - 22; ; --i) {
        if (ReadU32(tail + i) == kEocdSignature) {
            eocdPos = i;
            break;
        }
        if (i == 0) break;
    }
    if (eocdPos == std::numeric_limits<size_t>::max()) {
        if (error) *error = "ZIP end of central directory record not found: " + packagePath;
        return std::nullopt;
    }

    const uint8_t* eocd = tail + eocdPos;
    const uint16_t diskNumber = ReadU16(eocd + 4);
    const uint16_t centralDisk = ReadU16(eocd + 6);
    const uint16_t diskEntries = ReadU16(eocd + 8);
    const uint16_t totalEntries = ReadU16(eocd + 10);
    const uint32_t centralSize = ReadU32(eocd + 12);
    const uint32_t centralOffset = ReadU32(eocd + 16);

    if (diskNumber != 0 || centralDisk != 0 || diskEntries != totalEntries) {
        if (error) *error = "Multi-disk ZIP packages are not supported: " + packagePath;
        return std::nullopt;
    }
    if (centralOffset == 0xffffffffu || centralSize == 0xffffffffu) {
        if (error) *error = "ZIP64 packages are not supported: " + packagePath;
        return std::nullopt;
    }
    if (static_cast<uint64_t>(centralOffset) + centralSize > fileSize) {
        if (error) *error = "ZIP central directory is outside the package bounds: " + packagePath;
        return std::nullopt;
    }

    return ParseCentralDirectory(data + centralOffset,
                                 static_cast<size_t>(centralSize),
                                 totalEntries,
                                 packagePath,
                                 error);
}

std::optional<ZipDirectory> ReadCentralDirectoryFromFile(const std::string& filePath,
                                                         const std::string& packagePath,
                                                         std::string* error) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        if (error) *error = "Failed to open package: " + packagePath;
        return std::nullopt;
    }

    std::streamoff end = file.tellg();
    if (end < 22) {
        if (error) *error = "File is too small to be a ZIP package: " + packagePath;
        return std::nullopt;
    }
    const uint64_t fileSize = static_cast<uint64_t>(end);
    const uint64_t tailSize = std::min<uint64_t>(fileSize, 22u + 0xffffu);

    std::vector<uint8_t> tail(static_cast<size_t>(tailSize));
    file.seekg(static_cast<std::streamoff>(fileSize - tailSize), std::ios::beg);
    file.read(reinterpret_cast<char*>(tail.data()), static_cast<std::streamsize>(tail.size()));
    if (!file) {
        if (error) *error = "Failed to read ZIP end record from: " + packagePath;
        return std::nullopt;
    }

    size_t eocdPos = std::numeric_limits<size_t>::max();
    for (size_t i = tail.size() - 22; ; --i) {
        if (ReadU32(tail.data() + i) == kEocdSignature) {
            eocdPos = i;
            break;
        }
        if (i == 0) break;
    }
    if (eocdPos == std::numeric_limits<size_t>::max()) {
        if (error) *error = "ZIP end of central directory record not found: " + packagePath;
        return std::nullopt;
    }

    const uint8_t* eocd = tail.data() + eocdPos;
    const uint16_t diskNumber = ReadU16(eocd + 4);
    const uint16_t centralDisk = ReadU16(eocd + 6);
    const uint16_t diskEntries = ReadU16(eocd + 8);
    const uint16_t totalEntries = ReadU16(eocd + 10);
    const uint32_t centralSize = ReadU32(eocd + 12);
    const uint32_t centralOffset = ReadU32(eocd + 16);

    if (diskNumber != 0 || centralDisk != 0 || diskEntries != totalEntries) {
        if (error) *error = "Multi-disk ZIP packages are not supported: " + packagePath;
        return std::nullopt;
    }
    if (centralOffset == 0xffffffffu || centralSize == 0xffffffffu) {
        if (error) *error = "ZIP64 packages are not supported: " + packagePath;
        return std::nullopt;
    }
    if (static_cast<uint64_t>(centralOffset) + centralSize > fileSize) {
        if (error) *error = "ZIP central directory is outside the package bounds: " + packagePath;
        return std::nullopt;
    }

    std::vector<uint8_t> central(centralSize);
    file.seekg(static_cast<std::streamoff>(centralOffset), std::ios::beg);
    file.read(reinterpret_cast<char*>(central.data()), static_cast<std::streamsize>(central.size()));
    if (!file) {
        if (error) *error = "Failed to read ZIP central directory: " + packagePath;
        return std::nullopt;
    }

    auto dir = ParseCentralDirectory(central.data(),
                                     central.size(),
                                     totalEntries,
                                     packagePath,
                                     error);
    if (dir) dir->filePath = filePath;
    return dir;
}

std::optional<ZipDirectory> ReadCentralDirectory(const std::string& packagePath,
                                                 std::string* error) {
    ResolvedLocation packageLocation =
        ResolvedLocation::FromResolvedString(packagePath);
    std::string filePath;
    if (IsLocalFileResource(packageLocation, &filePath)) {
        return ReadCentralDirectoryFromFile(filePath, packagePath, error);
    }

    auto resource = ReadResource(packageLocation);
    if (!resource.success) {
        if (error) *error = std::move(resource.error);
        return std::nullopt;
    }

    auto data = std::make_shared<std::vector<uint8_t>>(std::move(resource.bytes));
    auto dir = ReadCentralDirectoryFromBuffer(data->data(),
                                              data->size(),
                                              packagePath,
                                              error);
    if (dir) dir->data = std::move(data);
    return dir;
}

const ZipEntry* FindEntry(const ZipDirectory& dir, const std::string& entryPath) {
    std::string normalized = NormalizePackageEntryPath(entryPath);
    for (const auto& entry : dir.entries) {
        if (entry.name == normalized) return &entry;
    }
    return nullptr;
}

bool ReadEntryData(const ZipDirectory& dir,
                   const ZipEntry& entry,
                   std::vector<uint8_t>& data,
                   std::string* error) {
    if (EndsWithSlash(entry.name)) {
        if (error) *error = "USDZ root layer entry is a directory: " + entry.name;
        return false;
    }
    if (entry.compressedSize != entry.uncompressedSize) {
        if (error) *error = "Stored ZIP entry has mismatched sizes: " + entry.name;
        return false;
    }

    if (dir.data) {
        const std::vector<uint8_t>& bytes = *dir.data;
        const uint64_t fileSize = static_cast<uint64_t>(bytes.size());
        if (static_cast<uint64_t>(entry.localHeaderOffset) + 30 > fileSize) {
            if (error) *error = "ZIP local header is outside the package bounds: " + entry.name;
            return false;
        }

        const uint8_t* local = bytes.data() + entry.localHeaderOffset;
        if (ReadU32(local) != kLocalSignature) {
            if (error) *error = "Malformed ZIP local header for entry: " + entry.name;
            return false;
        }

        const uint16_t nameLen = ReadU16(local + 26);
        const uint16_t extraLen = ReadU16(local + 28);
        const uint64_t dataOffset =
            static_cast<uint64_t>(entry.localHeaderOffset) + 30u + nameLen + extraLen;
        if (dataOffset + entry.compressedSize > fileSize) {
            if (error) *error = "ZIP entry data is outside the package bounds: " + entry.name;
            return false;
        }

        using Diff = std::vector<uint8_t>::difference_type;
        data.assign(bytes.begin() + static_cast<Diff>(dataOffset),
                    bytes.begin() + static_cast<Diff>(dataOffset + entry.uncompressedSize));
        return true;
    }

    const std::string& filePath = dir.filePath.empty() ? dir.packagePath : dir.filePath;
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        if (error) *error = "Failed to open package: " + dir.packagePath;
        return false;
    }

    const uint64_t fileSize = static_cast<uint64_t>(file.tellg());
    if (static_cast<uint64_t>(entry.localHeaderOffset) + 30 > fileSize) {
        if (error) *error = "ZIP local header is outside the package bounds: " + entry.name;
        return false;
    }

    uint8_t local[30] = {};
    file.seekg(static_cast<std::streamoff>(entry.localHeaderOffset), std::ios::beg);
    file.read(reinterpret_cast<char*>(local), sizeof(local));
    if (!file || ReadU32(local) != kLocalSignature) {
        if (error) *error = "Malformed ZIP local header for entry: " + entry.name;
        return false;
    }

    const uint16_t nameLen = ReadU16(local + 26);
    const uint16_t extraLen = ReadU16(local + 28);
    const uint64_t dataOffset =
        static_cast<uint64_t>(entry.localHeaderOffset) + 30u + nameLen + extraLen;
    if (dataOffset + entry.compressedSize > fileSize) {
        if (error) *error = "ZIP entry data is outside the package bounds: " + entry.name;
        return false;
    }

    data.resize(entry.uncompressedSize);
    file.seekg(static_cast<std::streamoff>(dataOffset), std::ios::beg);
    if (!data.empty()) {
        file.read(reinterpret_cast<char*>(data.data()),
                  static_cast<std::streamsize>(data.size()));
        if (!file) {
            if (error) *error = "Failed to read ZIP entry data: " + entry.name;
            return false;
        }
    }
    return true;
}

} // anonymous namespace

bool HasUsdzExtension(std::string_view path) {
    std::string packagePath;
    std::string entryPath;
    if (SplitPackageIdentifier(path, &packagePath, &entryPath))
        path = packagePath;

    size_t slash = path.find_last_of("/\\");
    size_t dot = path.find_last_of('.');
    if (dot == std::string_view::npos) return false;
    if (slash != std::string_view::npos && dot < slash) return false;
    return LowerAscii(path.substr(dot)) == ".usdz";
}

bool IsPackageIdentifier(std::string_view identifier) {
    std::string packagePath;
    std::string entryPath;
    return SplitPackageIdentifier(identifier, &packagePath, &entryPath);
}

bool SplitPackageIdentifier(std::string_view identifier,
                            std::string* packagePath,
                            std::string* entryPath) {
    if (identifier.empty() || identifier.back() != ']') return false;
    size_t open = std::string_view::npos;
    int bracketDepth = 0;
    for (size_t i = identifier.size(); i-- > 0;) {
        if (identifier[i] == ']') {
            ++bracketDepth;
        } else if (identifier[i] == '[') {
            --bracketDepth;
            if (bracketDepth == 0) {
                open = i;
                break;
            }
        }
    }
    if (open == std::string_view::npos || open == 0 || open + 1 >= identifier.size())
        return false;

    if (packagePath) packagePath->assign(identifier.substr(0, open));
    if (entryPath) {
        entryPath->assign(identifier.substr(open + 1,
                                            identifier.size() - open - 2));
        *entryPath = NormalizePackageEntryPath(*entryPath);
    }
    return true;
}

std::string NormalizePackageEntryPath(std::string_view path) {
    std::vector<std::string> parts;
    std::string current;

    auto flush = [&]() {
        if (current.empty() || current == ".") {
            current.clear();
            return;
        }
        if (current == "..") {
            if (!parts.empty()) parts.pop_back();
            current.clear();
            return;
        }
        parts.push_back(current);
        current.clear();
    };

    for (char c : path) {
        if (c == '/' || c == '\\') {
            flush();
        } else {
            current.push_back(c);
        }
    }
    flush();

    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out.push_back('/');
        out += parts[i];
    }
    return out;
}

std::string JoinPackageEntryPath(std::string_view anchorEntry,
                                 std::string_view relativePath) {
    std::string base(anchorEntry);
    auto slash = base.find_last_of('/');
    if (slash == std::string::npos) {
        base.clear();
    } else {
        base.erase(slash + 1);
    }
    base.append(relativePath);
    return NormalizePackageEntryPath(base);
}

std::string MakePackageIdentifier(const std::string& packagePath,
                                  const std::string& entryPath) {
    return packagePath + "[" + NormalizePackageEntryPath(entryPath) + "]";
}

std::string GetUsdzRootLayerPath(const std::string& packagePath,
                                 std::string* error) {
    std::string package;
    std::string entry;
    if (SplitPackageIdentifier(packagePath, &package, &entry)) return entry;

    auto dir = ReadCentralDirectory(packagePath, error);
    if (!dir) return {};
    if (dir->entries.empty()) {
        if (error) *error = "USDZ package has no entries: " + packagePath;
        return {};
    }
    if (EndsWithSlash(dir->entries.front().name)) {
        if (error) *error = "USDZ first central-directory entry is a directory: " +
                            dir->entries.front().name;
        return {};
    }
    return dir->entries.front().name;
}

UsdzReadResult ReadUsdzFile(const std::string& packagePathOrIdentifier) {
    UsdzReadResult result;

    std::string packagePath = packagePathOrIdentifier;
    std::string entryPath;
    if (SplitPackageIdentifier(packagePathOrIdentifier, &packagePath, &entryPath)) {
        result.packagePath = packagePath;
        result.entryPath = entryPath;
    } else {
        result.packagePath = packagePathOrIdentifier;
    }

    std::string error;
    auto dir = ReadCentralDirectory(packagePath, &error);
    if (!dir) {
        result.error = std::move(error);
        return result;
    }
    if (dir->entries.empty()) {
        result.error = "USDZ package has no entries: " + packagePath;
        return result;
    }

    const ZipEntry* entry = nullptr;
    if (entryPath.empty()) {
        entry = &dir->entries.front();
        result.entryPath = entry->name;
    } else {
        entry = FindEntry(*dir, entryPath);
        if (!entry) {
            result.error = "Entry not found in USDZ package: " + entryPath;
            return result;
        }
    }

    if (!ReadEntryData(*dir, *entry, result.data, &result.error))
        return result;

    result.success = true;
    return result;
}

UsdzWriteResult WriteUsdz(const std::vector<UsdzFileEntry>& entries) {
    UsdzWriteResult result;

    std::vector<UsdzFileEntry> normalized;
    if (!ValidateAndNormalizeEntries(entries, &normalized, &result.error))
        return result;

    std::vector<ZipCentralWriteEntry> central;
    central.reserve(normalized.size());

    for (const auto& entry : normalized) {
        ZipCentralWriteEntry centralEntry;
        if (!AppendLocalFileHeader(result.data, entry, &centralEntry, &result.error)) {
            result.data.clear();
            return result;
        }
        central.push_back(std::move(centralEntry));
    }

    if (!AppendCentralDirectory(result.data, central, &result.error)) {
        result.data.clear();
        return result;
    }

    result.success = true;
    return result;
}

UsdzWriteResult WriteUsdz(const Layer& rootLayer,
                          std::string_view rootEntryPath) {
    UsdzWriteResult result;
    std::vector<uint8_t> rootBytes =
        WriteRootLayerBytes(rootLayer, rootEntryPath, &result.error);
    if (rootBytes.empty()) return result;

    return WriteUsdz({UsdzFileEntry{
        NormalizePackageEntryPath(rootEntryPath),
        std::move(rootBytes),
    }});
}

bool WriteUsdzFile(const std::vector<UsdzFileEntry>& entries,
                   const ResolvedLocation& location,
                   std::string* error) {
    UsdzWriteResult package = WriteUsdz(entries);
    if (!package.success) {
        if (error) *error = std::move(package.error);
        return false;
    }

    auto write = WriteResource(location, package.data.data(), package.data.size());
    if (!write.success) {
        if (error) *error = std::move(write.error);
        return false;
    }
    return true;
}

bool WriteUsdzFile(const std::vector<UsdzFileEntry>& entries,
                   const std::string& path,
                   std::string* error) {
    return WriteUsdzFile(entries,
                         ResolvedLocation::FromResolvedString(path),
                         error);
}

bool WriteUsdzFile(const Layer& rootLayer,
                   const ResolvedLocation& location,
                   std::string_view rootEntryPath,
                   std::string* error) {
    std::vector<uint8_t> rootBytes =
        WriteRootLayerBytes(rootLayer, rootEntryPath, error);
    if (rootBytes.empty()) return false;

    return WriteUsdzFile({UsdzFileEntry{
                              NormalizePackageEntryPath(rootEntryPath),
                              std::move(rootBytes),
                          }},
                         location,
                         error);
}

bool WriteUsdzFile(const Layer& rootLayer,
                   const std::string& path,
                   std::string_view rootEntryPath,
                   std::string* error) {
    return WriteUsdzFile(rootLayer,
                         ResolvedLocation::FromResolvedString(path),
                         rootEntryPath,
                         error);
}

} // namespace nanousd
