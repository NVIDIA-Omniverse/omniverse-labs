// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <nanousd/nanousdapi.h>

#include <cctype>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

static std::string cstr(const char* value) {
    return value ? std::string(value) : std::string();
}

static bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

static bool contains(const std::string& value, const std::string& needle) {
    return value.find(needle) != std::string::npos;
}

static bool is_known_relationship_name(const std::string& name) {
    static constexpr const char* prefixes[] = {
        "material:binding",
        "physics:body",
        "physics:collisionGroup",
        "prototypes",
        "collection:",
        "proxyPrim",
        "skel:",
    };
    for (const char* prefix : prefixes) {
        if (starts_with(name, prefix) || name == prefix) {
            return true;
        }
    }
    return false;
}

static std::vector<std::string> split_sdf_path_elements(const std::string& text) {
    if (text.empty() || text == "/") {
        return {};
    }
    std::vector<std::string> out;
    std::string cur;
    int depth = 0;
    size_t start = text[0] == '/' ? 1 : 0;
    for (size_t i = start; i < text.size(); ++i) {
        char ch = text[i];
        if (ch == '[') {
            ++depth;
        } else if (ch == ']' && depth > 0) {
            --depth;
        }
        if (ch == '/' && depth == 0) {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
            continue;
        }
        cur.push_back(ch);
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

static int last_dot_outside_brackets(const std::string& text) {
    int depth = 0;
    for (size_t i = text.size(); i > 0; --i) {
        char ch = text[i - 1];
        if (ch == ']') {
            ++depth;
        } else if (ch == '[' && depth > 0) {
            --depth;
        } else if (ch == '.' && depth == 0) {
            return static_cast<int>(i - 1);
        }
    }
    return -1;
}

static int last_slash_outside_brackets(const std::string& text) {
    int depth = 0;
    for (size_t i = text.size(); i > 0; --i) {
        char ch = text[i - 1];
        if (ch == ']') {
            ++depth;
        } else if (ch == '[' && depth > 0) {
            --depth;
        } else if (ch == '/' && depth == 0) {
            return static_cast<int>(i - 1);
        }
    }
    return -1;
}

static bool ascii_isdigit(char ch) {
    return std::isdigit(static_cast<unsigned char>(ch)) != 0;
}

static int simple_sdf_path_validity(const std::string& text) {
    if (text.empty() || text == "/" || text == "." || text == "..") {
        return 1;
    }
    if (text.front() == '<' || text.back() == '>' || text.back() == '/') {
        return 0;
    }
    if (contains(text, "$") || contains(text, "//")) {
        return 0;
    }
    if (contains(text, ".") || contains(text, "[") || contains(text, "]")) {
        return -1;
    }
    if (ascii_isdigit(text.front())) {
        return 0;
    }

    bool at_segment_start = true;
    size_t start = text.front() == '/' ? 1 : 0;
    for (size_t i = start; i < text.size(); ++i) {
        char ch = text[i];
        if (ch == '/') {
            at_segment_start = true;
            continue;
        }
        if (at_segment_start) {
            if (ascii_isdigit(ch)) {
                return 0;
            }
            at_segment_start = false;
        }
        if (ch == ':') {
            return 0;
        }
    }
    return 1;
}

static bool sdf_path_is_valid_impl(const std::string& text) {
    int quick = simple_sdf_path_validity(text);
    if (quick != -1) {
        return quick == 1;
    }

    if (text.front() == '<' || text.back() == '>' || text.back() == '/') {
        return false;
    }
    if (contains(text, "$") || contains(text, "[]") || contains(text, "//")) {
        return false;
    }
    if (text.front() == '/') {
        for (const std::string& part : split_sdf_path_elements(text)) {
            if (!part.empty() && part.front() == '.') {
                return false;
            }
        }
    }
    if (ascii_isdigit(text.front())) {
        return false;
    }
    if (text != "." && text != "..") {
        for (const std::string& part : split_sdf_path_elements(text)) {
            if (!part.empty() && ascii_isdigit(part.front())) {
                return false;
            }
        }
    }
    if (contains(text, "[") || contains(text, "]")) {
        int depth = 0;
        for (char ch : text) {
            if (ch == '[') {
                ++depth;
            } else if (ch == ']') {
                --depth;
                if (depth < 0) {
                    return false;
                }
            }
        }
        if (depth != 0 || contains(text, "]/")) {
            return false;
        }
    }

    if (last_dot_outside_brackets(text) != -1) {
        int slash = last_slash_outside_brackets(text);
        std::string segment = text.substr(static_cast<size_t>(slash + 1));
        if (segment != "." && segment != "..") {
            int depth = 0;
            int dots = 0;
            for (char ch : segment) {
                if (ch == '[') {
                    ++depth;
                } else if (ch == ']' && depth > 0) {
                    --depth;
                } else if (ch == '.' && depth == 0) {
                    ++dots;
                }
            }
            if (dots > 2) {
                return false;
            }
            if (dots > 1 && !contains(segment, "[")) {
                return false;
            }
        }
    }

    int last_slash = last_slash_outside_brackets(text);
    if (last_slash != -1) {
        for (const std::string& segment : split_sdf_path_elements(text.substr(0, static_cast<size_t>(last_slash)))) {
            if (segment != "." && segment != ".." && last_dot_outside_brackets(segment) != -1) {
                return false;
            }
        }
    }

    for (const std::string& segment : split_sdf_path_elements(text)) {
        std::string prim_part = segment;
        int dot = last_dot_outside_brackets(prim_part);
        if (dot != -1) {
            prim_part = prim_part.substr(0, static_cast<size_t>(dot));
        }
        size_t bracket = prim_part.find('[');
        if (bracket != std::string::npos) {
            prim_part = prim_part.substr(0, bracket);
        }
        if (contains(prim_part, ":")) {
            return false;
        }
    }
    return true;
}

static std::string sdf_path_validated_text(const std::string& text) {
    return sdf_path_is_valid_impl(text) ? text : std::string();
}

static void require_size(size_t got, size_t expected, const std::string& what) {
    if (got != expected) {
        throw std::runtime_error(what + " expects " + std::to_string(expected) + " values");
    }
}

template <typename T>
static const T* data_or_null(const std::vector<T>& values) {
    return values.empty() ? nullptr : values.data();
}

static std::vector<const char*> string_ptrs(const std::vector<std::string>& values) {
    std::vector<const char*> out;
    out.reserve(values.size());
    for (const std::string& value : values) {
        out.push_back(value.c_str());
    }
    return out;
}

static std::string owned_string(const char* value) {
    if (!value) {
        return {};
    }
    std::string out(value);
    nanousd_free_string(value);
    return out;
}

template <typename T>
static nb::object vector_to_numpy(std::vector<T>&& values, int components = 1) {
    int cols = components > 0 ? components : 1;
    if (cols > 1 && values.size() % static_cast<size_t>(cols) != 0) {
        return nb::none();
    }

    auto* storage = new std::vector<T>(std::move(values));
    nb::capsule owner(storage, [](void* ptr) noexcept {
        delete static_cast<std::vector<T>*>(ptr);
    });

    if (cols <= 1) {
        return nb::ndarray<nb::numpy, T>(
            storage->data(), {storage->size()}, owner).cast();
    }

    return nb::ndarray<nb::numpy, T>(
        storage->data(),
        {storage->size() / static_cast<size_t>(cols), static_cast<size_t>(cols)},
        owner).cast();
}

static nb::object fan_triangulate_indices_numpy(
        nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>, nb::c_contig> counts,
        nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>, nb::c_contig> indices,
        bool flip_winding = false) {
    const int32_t* counts_data = counts.data();
    const int32_t* indices_data = indices.data();
    size_t face_count = counts.shape(0);
    size_t index_count = indices.shape(0);

    int64_t num_tris = 0;
    int64_t cursor = 0;
    for (size_t i = 0; i < face_count; ++i) {
        int32_t n = counts_data[i];
        if (n < 0) {
            throw std::runtime_error("face vertex counts must be non-negative");
        }
        cursor += n;
        if (cursor > static_cast<int64_t>(index_count)) {
            throw std::runtime_error("face vertex counts exceed index array length");
        }
        if (n > 2) {
            num_tris += static_cast<int64_t>(n) - 2;
        }
    }

    std::vector<int32_t> out(static_cast<size_t>(num_tris) * 3);
    int64_t face_base = 0;
    int64_t tri = 0;
    for (size_t face = 0; face < face_count; ++face) {
        int32_t n = counts_data[face];
        if (n > 2) {
            int32_t anchor = indices_data[face_base];
            for (int32_t local = 0; local < n - 2; ++local) {
                int32_t b = indices_data[face_base + local + 1];
                int32_t c = indices_data[face_base + local + 2];
                size_t dst = static_cast<size_t>(tri) * 3;
                if (flip_winding) {
                    out[dst] = c;
                    out[dst + 1] = b;
                    out[dst + 2] = anchor;
                } else {
                    out[dst] = anchor;
                    out[dst + 1] = b;
                    out[dst + 2] = c;
                }
                ++tri;
            }
        }
        face_base += n;
    }
    return vector_to_numpy(std::move(out), 3);
}

static size_t skip_ws(const std::string& text, size_t pos, size_t end) {
    while (pos < end && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    return pos;
}

static std::optional<std::string> find_usda_layer_metadata_string(
    const std::string& usda, const std::string& key) {
    size_t marker = usda.find("#usda");
    if (marker == std::string::npos) {
        return std::nullopt;
    }
    size_t open = usda.find('(', marker);
    if (open == std::string::npos) {
        return std::nullopt;
    }
    size_t close = usda.find(')', open + 1);
    if (close == std::string::npos) {
        return std::nullopt;
    }

    size_t line_start = open + 1;
    while (line_start < close) {
        size_t line_end = usda.find('\n', line_start);
        if (line_end == std::string::npos || line_end > close) {
            line_end = close;
        }

        size_t pos = skip_ws(usda, line_start, line_end);
        if (pos + key.size() <= line_end && usda.compare(pos, key.size(), key) == 0) {
            pos = skip_ws(usda, pos + key.size(), line_end);
            if (pos < line_end && usda[pos] == '=') {
                pos = skip_ws(usda, pos + 1, line_end);
                if (pos < line_end && usda[pos] == '"') {
                    size_t value_end = usda.find('"', pos + 1);
                    if (value_end != std::string::npos && value_end <= line_end) {
                        return usda.substr(pos + 1, value_end - pos - 1);
                    }
                }
            }
        }

        line_start = line_end + 1;
    }
    return std::nullopt;
}

struct StageState {
    NanousdStage stage = nullptr;

    ~StageState() {
        if (stage) {
            nanousd_close(stage);
            stage = nullptr;
        }
    }

    StageState() = default;
    StageState(const StageState&) = delete;
    StageState& operator=(const StageState&) = delete;
};

struct PrimHandle {
    std::shared_ptr<StageState> stage;
    NanousdPrim prim = nullptr;

    PrimHandle(std::shared_ptr<StageState> stage_, NanousdPrim prim_)
        : stage(std::move(stage_)), prim(prim_) {}

    ~PrimHandle() {
        if (prim) {
            nanousd_freeprim(prim);
            prim = nullptr;
        }
    }

    PrimHandle(const PrimHandle&) = delete;
    PrimHandle& operator=(const PrimHandle&) = delete;
};

struct PathHandle {
    NanousdPath path = nullptr;

    explicit PathHandle(NanousdPath path_) : path(path_) {}

    ~PathHandle() {
        if (path) {
            nanousd_path_free(path);
            path = nullptr;
        }
    }

    PathHandle(const PathHandle&) = delete;
    PathHandle& operator=(const PathHandle&) = delete;
};

class Path {
public:
    Path() = default;

    explicit Path(const std::string& text) {
        NanousdPath path = nanousd_path_parse(text.c_str());
        if (!path) {
            throw std::runtime_error("nanousd_path_parse failed");
        }
        handle_ = std::make_shared<PathHandle>(path);
    }

    explicit Path(NanousdPath path) {
        if (path) {
            handle_ = std::make_shared<PathHandle>(path);
        }
    }

    static Path parse(const std::string& text) { return Path(text); }

    bool valid() const { return handle_ && handle_->path; }

    std::string str() const { return cstr(nanousd_path_str(require())); }
    std::string name() const { return cstr(nanousd_path_name(require())); }

    bool is_absolute() const { return nanousd_path_is_absolute(require()) != 0; }
    bool is_root() const { return nanousd_path_is_root(require()) != 0; }
    bool is_property() const { return nanousd_path_is_property(require()) != 0; }

    Path append_child(const std::string& child) const {
        NanousdPath path = nanousd_path_append_child(require(), child.c_str());
        if (!path) {
            throw std::runtime_error("nanousd_path_append_child failed");
        }
        return Path(path);
    }

    Path append_property(const std::string& prop) const {
        NanousdPath path = nanousd_path_append_property(require(), prop.c_str());
        if (!path) {
            throw std::runtime_error("nanousd_path_append_property failed");
        }
        return Path(path);
    }

    std::optional<Path> parent() const {
        NanousdPath path = nanousd_path_parent(require());
        if (!path) {
            return std::nullopt;
        }
        return Path(path);
    }

    bool equal(const Path& other) const {
        return nanousd_path_equal(require(), other.require()) != 0;
    }

private:
    std::shared_ptr<PathHandle> handle_;

    NanousdPath require() const {
        if (!handle_ || !handle_->path) {
            throw std::runtime_error("invalid nanousd Path");
        }
        return handle_->path;
    }
};

struct ListOpHandle {
    NanousdListOp op = nullptr;

    explicit ListOpHandle(NanousdListOp op_) : op(op_) {}

    ~ListOpHandle() {
        if (op) {
            nanousd_listop_free(op);
            op = nullptr;
        }
    }

    ListOpHandle(const ListOpHandle&) = delete;
    ListOpHandle& operator=(const ListOpHandle&) = delete;
};

class ListOp {
public:
    ListOp() = default;

    explicit ListOp(NanousdListOp op) {
        if (op) {
            handle_ = std::make_shared<ListOpHandle>(op);
        }
    }

    static ListOp create_explicit(const std::vector<std::string>& items) {
        std::vector<const char*> ptrs = string_ptrs(items);
        NanousdListOp op = nanousd_listop_create_explicit(ptrs.data(), static_cast<int>(ptrs.size()));
        if (!op) {
            throw std::runtime_error("nanousd_listop_create_explicit failed");
        }
        return ListOp(op);
    }

    static ListOp create(const std::vector<std::string>& prepended = {},
                         const std::vector<std::string>& appended = {},
                         const std::vector<std::string>& deleted = {}) {
        std::vector<const char*> prepended_ptrs = string_ptrs(prepended);
        std::vector<const char*> appended_ptrs = string_ptrs(appended);
        std::vector<const char*> deleted_ptrs = string_ptrs(deleted);
        NanousdListOp op = nanousd_listop_create(
            prepended_ptrs.data(), static_cast<int>(prepended_ptrs.size()),
            appended_ptrs.data(), static_cast<int>(appended_ptrs.size()),
            deleted_ptrs.data(), static_cast<int>(deleted_ptrs.size()));
        if (!op) {
            throw std::runtime_error("nanousd_listop_create failed");
        }
        return ListOp(op);
    }

    static ListOp combine(const ListOp& stronger, const ListOp& weaker) {
        NanousdListOp op = nanousd_listop_combine(stronger.require(), weaker.require());
        if (!op) {
            throw std::runtime_error("nanousd_listop_combine failed");
        }
        return ListOp(op);
    }

    bool valid() const { return handle_ && handle_->op; }
    bool is_explicit() const { return nanousd_listop_is_explicit(require()) != 0; }

    std::vector<std::string> items() const {
        return collect(nanousd_listop_nitems, nanousd_listop_item);
    }

    std::vector<std::string> prepended_items() const {
        return collect(nanousd_listop_nprepended, nanousd_listop_prepended);
    }

    std::vector<std::string> appended_items() const {
        return collect(nanousd_listop_nappended, nanousd_listop_appended);
    }

    std::vector<std::string> deleted_items() const {
        return collect(nanousd_listop_ndeleted, nanousd_listop_deleted);
    }

private:
    using CountFn = int (*)(NanousdListOp);
    using ItemFn = const char* (*)(NanousdListOp, int);

    std::shared_ptr<ListOpHandle> handle_;

    NanousdListOp require() const {
        if (!handle_ || !handle_->op) {
            throw std::runtime_error("invalid nanousd ListOp");
        }
        return handle_->op;
    }

    std::vector<std::string> collect(CountFn count_fn, ItemFn item_fn) const {
        NanousdListOp op = require();
        std::vector<std::string> out;
        int n = count_fn(op);
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            out.push_back(cstr(item_fn(op, i)));
        }
        return out;
    }
};

struct Diagnostic {
    int severity = 0;
    int category = 0;
    std::string message;
    std::string prim_path;
    std::string layer_path;
    std::string asset_path;
    int arc_type = 0;
};

class Prim {
public:
    Prim() = default;
    explicit Prim(std::shared_ptr<PrimHandle> handle) : handle_(std::move(handle)) {}

    bool valid() const {
        return handle_ && handle_->prim && nanousd_prim_isvalid(handle_->prim);
    }

    std::string path() const { return cstr(nanousd_path(require())); }
    std::string name() const { return cstr(nanousd_name(require())); }
    std::string type_name() const { return cstr(nanousd_typename(require())); }
    std::string kind() const { return cstr(nanousd_kind(require())); }

    bool is_active() const { return nanousd_isactive(require()) != 0; }
    bool is_defined() const { return nanousd_isdefined(require()) != 0; }
    bool is_abstract() const { return nanousd_isabstract(require()) != 0; }
    bool is_instanceable() const { return nanousd_isinstanceable(require()) != 0; }
    bool is_instance() const { return nanousd_isinstance(require()) != 0; }
    bool is_prototype() const { return nanousd_isprototype(require()) != 0; }
    bool is_in_prototype() const { return nanousd_isinprototype(require()) != 0; }

    bool is_a(const std::string& type_name) const {
        return nanousd_isa(require(), type_name.c_str()) != 0;
    }

    bool has_api(const std::string& api_name) const {
        return nanousd_hasapi(require(), api_name.c_str()) != 0;
    }

    bool apply_api(const std::string& api_name) const {
        return nanousd_apply_api(require(), api_name.c_str()) != 0;
    }

    bool remove_api(const std::string& api_name) const {
        return nanousd_remove_api(require(), api_name.c_str()) != 0;
    }

    bool set_active(bool active) const {
        return nanousd_set_active(require(), active ? 1 : 0) != 0;
    }

    bool set_instanceable(bool instanceable) const {
        return nanousd_set_instanceable(require(), instanceable ? 1 : 0) != 0;
    }

    bool set_specifier(const std::string& specifier) const {
        return nanousd_set_specifier(require(), specifier.c_str()) != 0;
    }

    bool remove() {
        NanousdPrim p = require();
        if (!nanousd_remove_prim(p)) {
            return false;
        }
        handle_->prim = nullptr;
        return true;
    }

    std::vector<Prim> children() const {
        NanousdPrim p = require();
        std::vector<Prim> out;
        int n = nanousd_nchildren(p);
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            NanousdPrim child = nanousd_child(p, i);
            if (child) {
                out.emplace_back(make_prim(child));
            }
        }
        return out;
    }

    std::optional<Prim> child(const std::string& name) const {
        NanousdPrim child = nanousd_childname(require(), name.c_str());
        if (!child) {
            return std::nullopt;
        }
        return make_prim(child);
    }

    std::optional<Prim> parent() const {
        NanousdPrim parent = nanousd_parent(require());
        if (!parent) {
            return std::nullopt;
        }
        return make_prim(parent);
    }

    std::optional<Prim> prototype() const {
        NanousdPrim prototype = nanousd_prototype(require());
        if (!prototype) {
            return std::nullopt;
        }
        return make_prim(prototype);
    }

    std::vector<Prim> instances() const {
        NanousdPrim p = require();
        std::vector<Prim> out;
        int n = nanousd_ninstances(p);
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            NanousdPrim instance = nanousd_instance(p, i);
            if (instance) {
                out.emplace_back(make_prim(instance));
            }
        }
        return out;
    }

    std::vector<std::string> attribute_names() const {
        NanousdPrim p = require();
        std::vector<std::string> out;
        int n = nanousd_nattribs(p);
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            out.push_back(cstr(nanousd_attribname(p, i)));
        }
        return out;
    }

    std::vector<std::string> attribute_names_in_namespace(const std::string& namespace_name) const {
        std::string prefix = namespace_name;
        if (!ends_with(prefix, ":")) {
            prefix.push_back(':');
        }

        NanousdPrim p = require();
        std::vector<std::string> out;
        int n = nanousd_nattribs(p);
        for (int i = 0; i < n; ++i) {
            const char* raw_name = nanousd_attribname(p, i);
            if (!raw_name) {
                continue;
            }
            std::string name(raw_name);
            if (name.compare(0, prefix.size(), prefix) == 0) {
                out.push_back(std::move(name));
            }
        }
        return out;
    }

    std::vector<std::string> authored_attribute_names() const {
        NanousdPrim p = require();
        std::vector<std::string> out;
        int n = nanousd_nauthored_attribs(p);
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            out.push_back(cstr(nanousd_authored_attribname(p, i)));
        }
        return out;
    }

    std::vector<std::string> authored_attribute_names_in_namespace(const std::string& namespace_name) const {
        std::string prefix = namespace_name;
        if (!ends_with(prefix, ":")) {
            prefix.push_back(':');
        }

        NanousdPrim p = require();
        std::vector<std::string> out;
        int n = nanousd_nauthored_attribs(p);
        for (int i = 0; i < n; ++i) {
            const char* raw_name = nanousd_authored_attribname(p, i);
            if (!raw_name) {
                continue;
            }
            std::string name(raw_name);
            if (name.compare(0, prefix.size(), prefix) == 0 &&
                nanousd_attrib_authored(p, raw_name) != 0) {
                out.push_back(std::move(name));
            }
        }
        return out;
    }

    bool has_attribute(const std::string& name) const {
        return nanousd_hasattrib(require(), name.c_str()) != 0;
    }

    bool is_attribute_authored(const std::string& name) const {
        return nanousd_attrib_authored(require(), name.c_str()) != 0;
    }

    std::string attribute_type(const std::string& name) const {
        return cstr(nanousd_attribtype(require(), name.c_str()));
    }

    std::string attribute_interpolation(const std::string& name) const {
        return cstr(nanousd_attrib_interpolation(require(), name.c_str()));
    }

    bool create_attribute(const std::string& name, const std::string& type_name) const {
        return nanousd_create_attrib(require(), name.c_str(), type_name.c_str()) != 0;
    }

    std::optional<double> read_double(const std::string& name) const {
        int ok = 0;
        double value = nanousd_attribd(require(), name.c_str(), &ok);
        if (!ok) {
            return std::nullopt;
        }
        return value;
    }

    std::optional<float> read_float(const std::string& name) const {
        int ok = 0;
        float value = nanousd_attribf(require(), name.c_str(), &ok);
        if (!ok) {
            return std::nullopt;
        }
        return value;
    }

    std::optional<int> read_int(const std::string& name) const {
        int ok = 0;
        int value = nanousd_attribi(require(), name.c_str(), &ok);
        if (!ok) {
            return std::nullopt;
        }
        return value;
    }

    std::optional<int64_t> read_int64(const std::string& name) const {
        int ok = 0;
        int64_t value = nanousd_attribi64(require(), name.c_str(), &ok);
        if (!ok) {
            return std::nullopt;
        }
        return value;
    }

    std::optional<bool> read_bool(const std::string& name) const {
        int ok = 0;
        int value = nanousd_attribb(require(), name.c_str(), &ok);
        if (!ok) {
            return std::nullopt;
        }
        return value != 0;
    }

    std::optional<std::string> read_string(const std::string& name) const {
        int ok = 0;
        const char* value = nanousd_attribs(require(), name.c_str(), &ok);
        if (!ok || !value) {
            return std::nullopt;
        }
        return std::string(value);
    }

    std::optional<std::string> read_token(const std::string& name) const {
        int ok = 0;
        const char* value = nanousd_attrib_token(require(), name.c_str(), &ok);
        if (!ok || !value) {
            return std::nullopt;
        }
        return std::string(value);
    }

    std::optional<std::string> read_asset(const std::string& name) const {
        int ok = 0;
        const char* value = nanousd_attribasset(require(), name.c_str(), &ok);
        if (!ok || !value) {
            return std::nullopt;
        }
        return std::string(value);
    }

    std::optional<std::vector<float>> read_vec2f(const std::string& name) const {
        float v[2] = {};
        if (!nanousd_attribv2f(require(), name.c_str(), v)) {
            return std::nullopt;
        }
        return std::vector<float>{v[0], v[1]};
    }

    std::optional<std::vector<float>> read_vec3f(const std::string& name) const {
        float v[3] = {};
        if (!nanousd_attribv3f(require(), name.c_str(), v)) {
            return std::nullopt;
        }
        return std::vector<float>{v[0], v[1], v[2]};
    }

    std::optional<std::vector<float>> read_vec4f(const std::string& name) const {
        float v[4] = {};
        if (!nanousd_attribv4f(require(), name.c_str(), v)) {
            return std::nullopt;
        }
        return std::vector<float>{v[0], v[1], v[2], v[3]};
    }

    std::optional<std::vector<double>> read_vec2d(const std::string& name) const {
        double v[2] = {};
        if (!nanousd_attribv2d(require(), name.c_str(), v)) {
            return std::nullopt;
        }
        return std::vector<double>{v[0], v[1]};
    }

    std::optional<std::vector<double>> read_vec3d(const std::string& name) const {
        double v[3] = {};
        if (!nanousd_attribv3d(require(), name.c_str(), v)) {
            return std::nullopt;
        }
        return std::vector<double>{v[0], v[1], v[2]};
    }

    std::optional<std::vector<double>> read_vec4d(const std::string& name) const {
        double v[4] = {};
        if (!nanousd_attribv4d(require(), name.c_str(), v)) {
            return std::nullopt;
        }
        return std::vector<double>{v[0], v[1], v[2], v[3]};
    }

    std::optional<std::vector<int>> read_vec2i(const std::string& name) const {
        int v[2] = {};
        if (!nanousd_attribv2i(require(), name.c_str(), v)) {
            return std::nullopt;
        }
        return std::vector<int>{v[0], v[1]};
    }

    std::optional<std::vector<int>> read_vec3i(const std::string& name) const {
        int v[3] = {};
        if (!nanousd_attribv3i(require(), name.c_str(), v)) {
            return std::nullopt;
        }
        return std::vector<int>{v[0], v[1], v[2]};
    }

    std::optional<std::vector<int>> read_vec4i(const std::string& name) const {
        int v[4] = {};
        if (!nanousd_attribv4i(require(), name.c_str(), v)) {
            return std::nullopt;
        }
        return std::vector<int>{v[0], v[1], v[2], v[3]};
    }

    std::optional<std::vector<float>> read_quatf(const std::string& name) const {
        float q[4] = {};
        if (!nanousd_attribqf(require(), name.c_str(), q)) {
            return std::nullopt;
        }
        return std::vector<float>{q[0], q[1], q[2], q[3]};
    }

    std::optional<std::vector<double>> read_quatd(const std::string& name) const {
        double q[4] = {};
        if (!nanousd_attribqd(require(), name.c_str(), q)) {
            return std::nullopt;
        }
        return std::vector<double>{q[0], q[1], q[2], q[3]};
    }

    std::optional<std::vector<double>> read_matrix3d(const std::string& name) const {
        double m[9] = {};
        if (!nanousd_attribm3d(require(), name.c_str(), m)) {
            return std::nullopt;
        }
        return std::vector<double>(m, m + 9);
    }

    std::optional<std::vector<double>> read_matrix4d(const std::string& name) const {
        double m[16] = {};
        if (!nanousd_attribm4d(require(), name.c_str(), m)) {
            return std::nullopt;
        }
        return std::vector<double>(m, m + 16);
    }

    nb::object get_attribute(const std::string& name) const {
        std::string type = attribute_type(name);
        if (type == "double") {
            auto value = read_double(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "float") {
            auto value = read_float(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "int") {
            auto value = read_int(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "int64") {
            auto value = read_int64(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "bool") {
            auto value = read_bool(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "token") {
            auto value = read_token(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "asset") {
            auto value = read_asset(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "string") {
            auto value = read_string(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (ends_with(type, "[]")) {
            if (type == "float[]") {
                return nb::cast(read_float_array(name));
            }
            if (type == "double[]") {
                return nb::cast(read_double_array(name));
            }
            if (type == "int[]") {
                return nb::cast(read_int_array(name));
            }
            if (type == "int64[]") {
                return nb::cast(read_int64_array(name));
            }
            if (type == "string[]" || type == "asset[]") {
                return nb::cast(read_string_array(name));
            }
            if (type == "token[]") {
                return nb::cast(read_token_array(name));
            }
            if (contains(type, "3f[]")) {
                return nb::cast(read_vec3f_array(name));
            }
            if (contains(type, "3d[]")) {
                return nb::cast(read_vec3d_array(name));
            }
        }
        if (type == "float2" || type == "texCoord2f" || contains(type, "2f")) {
            auto value = read_vec2f(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "float3" || type == "point3f" || type == "normal3f" ||
            type == "color3f" || type == "vector3f" || contains(type, "3f")) {
            auto value = read_vec3f(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "float4" || type == "color4f" || contains(type, "4f")) {
            auto value = read_vec4f(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "double2" || type == "texCoord2d" || contains(type, "2d")) {
            auto value = read_vec2d(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "double3" || type == "point3d" || type == "normal3d" ||
            type == "color3d" || type == "vector3d" || contains(type, "3d")) {
            auto value = read_vec3d(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "double4" || type == "color4d" || contains(type, "4d")) {
            auto value = read_vec4d(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "int2" || contains(type, "2i")) {
            auto value = read_vec2i(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "int3" || contains(type, "3i")) {
            auto value = read_vec3i(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "int4" || contains(type, "4i")) {
            auto value = read_vec4i(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "quatf") {
            auto value = read_quatf(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "quatd") {
            auto value = read_quatd(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "matrix3d") {
            auto value = read_matrix3d(name);
            return value ? nb::cast(*value) : nb::none();
        }
        if (type == "matrix4d") {
            auto value = read_matrix4d(name);
            return value ? nb::cast(*value) : nb::none();
        }
        return nb::none();
    }

    nb::dict authored_attribute_values_in_namespace(const std::string& namespace_name) const {
        std::string prefix = namespace_name;
        if (!ends_with(prefix, ":")) {
            prefix.push_back(':');
        }

        NanousdPrim p = require();
        nb::dict out;
        int n = nanousd_nauthored_attribs(p);
        for (int i = 0; i < n; ++i) {
            const char* raw_name = nanousd_authored_attribname(p, i);
            if (!raw_name) {
                continue;
            }
            std::string name(raw_name);
            if (name.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            nb::object value = get_attribute(name);
            if (!value.is_none()) {
                out[nb::str(name.c_str())] = value;
            }
        }
        return out;
    }

    nb::dict authored_attribute_values_in_namespaces(const std::vector<std::string>& namespace_names) const {
        std::vector<std::string> prefixes;
        prefixes.reserve(namespace_names.size());
        for (const auto& namespace_name : namespace_names) {
            std::string prefix = namespace_name;
            if (!ends_with(prefix, ":")) {
                prefix.push_back(':');
            }
            prefixes.push_back(std::move(prefix));
        }

        NanousdPrim p = require();
        nb::dict out;
        int n = nanousd_nauthored_attribs(p);
        for (int i = 0; i < n; ++i) {
            const char* raw_name = nanousd_authored_attribname(p, i);
            if (!raw_name) {
                continue;
            }
            std::string name(raw_name);
            bool matched = false;
            for (const auto& prefix : prefixes) {
                if (name.compare(0, prefix.size(), prefix) == 0) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                continue;
            }
            nb::object value = get_attribute(name);
            if (!value.is_none()) {
                out[nb::str(name.c_str())] = value;
            }
        }
        return out;
    }

    nb::dict authored_attribute_values(const std::vector<std::string>& names) const {
        NanousdPrim p = require();
        std::unordered_set<std::string> requested(names.begin(), names.end());
        nb::dict out;
        int n = nanousd_nauthored_attribs(p);
        for (int i = 0; i < n; ++i) {
            const char* raw_name = nanousd_authored_attribname(p, i);
            if (!raw_name || requested.find(raw_name) == requested.end()) {
                continue;
            }
            std::string name(raw_name);
            nb::object value = get_attribute(name);
            if (!value.is_none()) {
                out[nb::str(name.c_str())] = value;
            }
        }
        return out;
    }

    std::optional<std::vector<double>> local_transform() const {
        return local_transform_at(std::numeric_limits<double>::quiet_NaN());
    }

    std::optional<std::vector<double>> local_transform_at(double time) const {
        double m[16] = {};
        int reset = 0;
        if (!nanousd_get_local_transform(require(), time, m, &reset)) {
            return std::nullopt;
        }
        return std::vector<double>(m, m + 16);
    }

    nb::tuple local_transform_info_at(double time) const {
        double m[16] = {};
        int reset = 0;
        if (!nanousd_get_local_transform(require(), time, m, &reset)) {
            return nb::make_tuple(nb::none(), false);
        }
        return nb::make_tuple(std::vector<double>(m, m + 16), reset != 0);
    }

    int array_len(const std::string& name) const {
        return nanousd_attribarraylen(require(), name.c_str());
    }

    std::vector<float> read_float_array_flat(const std::string& name, int components = 1) const {
        return read_numeric_array_flat<float>(name, nanousd_attribarrayf, components);
    }

    std::vector<double> read_double_array_flat(const std::string& name, int components = 1) const {
        return read_numeric_array_flat<double>(name, nanousd_attribarrayd, components);
    }

    std::vector<int> read_int_array_flat(const std::string& name, int components = 1) const {
        return read_numeric_array_flat<int>(name, nanousd_attribarrayi, components);
    }

    nb::object read_float_array_numpy(const std::string& name, int components = 1) const {
        return read_numeric_array_numpy<float>(name, nanousd_attribarrayf, components);
    }

    nb::object read_double_array_numpy(const std::string& name, int components = 1) const {
        return read_numeric_array_numpy<double>(name, nanousd_attribarrayd, components);
    }

    nb::object read_int_array_numpy(const std::string& name, int components = 1) const {
        return read_numeric_array_numpy<int>(name, nanousd_attribarrayi, components);
    }

    nb::object read_int64_array_numpy(const std::string& name) const {
        return read_numeric_array_numpy<int64_t>(name, nanousd_attribarrayi64, 1);
    }

    std::vector<float> read_float_array(const std::string& name) const {
        return read_numeric_array<float>(name, nanousd_attribarrayf);
    }

    std::vector<double> read_double_array(const std::string& name) const {
        return read_numeric_array<double>(name, nanousd_attribarrayd);
    }

    std::vector<int> read_int_array(const std::string& name) const {
        return read_numeric_array<int>(name, nanousd_attribarrayi);
    }

    std::vector<int64_t> read_int64_array(const std::string& name) const {
        return read_numeric_array<int64_t>(name, nanousd_attribarrayi64);
    }

    std::vector<float> read_vec3f_array(const std::string& name) const {
        NanousdPrim p = require();
        int n = nanousd_attribarraylen(p, name.c_str());
        if (n <= 0) {
            return {};
        }
        std::vector<float> out(static_cast<size_t>(n) * 3);
        int written = nanousd_attribarrayv3f(p, name.c_str(), out.data(), n);
        if (written < 0) {
            return {};
        }
        out.resize(static_cast<size_t>(written) * 3);
        return out;
    }

    std::vector<double> read_vec3d_array(const std::string& name) const {
        NanousdPrim p = require();
        int n = nanousd_attribarraylen(p, name.c_str());
        if (n <= 0) {
            return {};
        }
        std::vector<double> out(static_cast<size_t>(n) * 3);
        int written = nanousd_attribarrayv3d(p, name.c_str(), out.data(), n);
        if (written < 0) {
            return {};
        }
        out.resize(static_cast<size_t>(written) * 3);
        return out;
    }

    std::vector<std::string> read_string_array(const std::string& name) const {
        NanousdPrim p = require();
        std::vector<std::string> out;
        int n = nanousd_attribarrays_len(p, name.c_str());
        if (n <= 0) {
            return {};
        }
        out.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            out.push_back(cstr(nanousd_attribarrays(p, name.c_str(), i)));
        }
        return out;
    }

    std::vector<std::string> read_token_array(const std::string& name) const {
        NanousdPrim p = require();
        std::vector<std::string> out;
        int n = nanousd_attribarraytokens_len(p, name.c_str());
        if (n <= 0) {
            return {};
        }
        out.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            out.push_back(cstr(nanousd_attribarraytokens(p, name.c_str(), i)));
        }
        return out;
    }

    bool set_double(const std::string& name, double value) const {
        return nanousd_set_attribd(require(), name.c_str(), value) != 0;
    }

    bool set_float(const std::string& name, float value) const {
        return nanousd_set_attribf(require(), name.c_str(), value) != 0;
    }

    bool set_int(const std::string& name, int value) const {
        return nanousd_set_attribi(require(), name.c_str(), value) != 0;
    }

    bool set_int64(const std::string& name, int64_t value) const {
        return nanousd_set_attribi64(require(), name.c_str(), value) != 0;
    }

    bool set_bool(const std::string& name, bool value) const {
        return nanousd_set_attribb(require(), name.c_str(), value ? 1 : 0) != 0;
    }

    bool set_string(const std::string& name, const std::string& value) const {
        return nanousd_set_attribs(require(), name.c_str(), value.c_str()) != 0;
    }

    bool set_token(const std::string& name, const std::string& value) const {
        return nanousd_set_attrib_token(require(), name.c_str(), value.c_str()) != 0;
    }

    bool set_asset(const std::string& name, const std::string& value) const {
        return nanousd_set_attrib_asset(require(), name.c_str(), value.c_str()) != 0;
    }

    bool set_vec2f(const std::string& name, const std::vector<float>& value) const {
        require_size(value.size(), 2, name);
        return nanousd_set_attribv2f(require(), name.c_str(), value.data()) != 0;
    }

    bool set_vec3f(const std::string& name, const std::vector<float>& value) const {
        require_size(value.size(), 3, name);
        return nanousd_set_attribv3f(require(), name.c_str(), value.data()) != 0;
    }

    bool set_vec4f(const std::string& name, const std::vector<float>& value) const {
        require_size(value.size(), 4, name);
        return nanousd_set_attribv4f(require(), name.c_str(), value.data()) != 0;
    }

    bool set_vec2d(const std::string& name, const std::vector<double>& value) const {
        require_size(value.size(), 2, name);
        return nanousd_set_attribv2d(require(), name.c_str(), value.data()) != 0;
    }

    bool set_vec3d(const std::string& name, const std::vector<double>& value) const {
        require_size(value.size(), 3, name);
        return nanousd_set_attribv3d(require(), name.c_str(), value.data()) != 0;
    }

    bool set_vec4d(const std::string& name, const std::vector<double>& value) const {
        require_size(value.size(), 4, name);
        return nanousd_set_attribv4d(require(), name.c_str(), value.data()) != 0;
    }

    bool set_vec2i(const std::string& name, const std::vector<int>& value) const {
        require_size(value.size(), 2, name);
        return nanousd_set_attribv2i(require(), name.c_str(), value.data()) != 0;
    }

    bool set_vec3i(const std::string& name, const std::vector<int>& value) const {
        require_size(value.size(), 3, name);
        return nanousd_set_attribv3i(require(), name.c_str(), value.data()) != 0;
    }

    bool set_vec4i(const std::string& name, const std::vector<int>& value) const {
        require_size(value.size(), 4, name);
        return nanousd_set_attribv4i(require(), name.c_str(), value.data()) != 0;
    }

    bool set_quatf(const std::string& name, const std::vector<float>& value) const {
        require_size(value.size(), 4, name);
        return nanousd_set_attribqf(require(), name.c_str(), value.data()) != 0;
    }

    bool set_quatd(const std::string& name, const std::vector<double>& value) const {
        require_size(value.size(), 4, name);
        return nanousd_set_attribqd(require(), name.c_str(), value.data()) != 0;
    }

    bool set_matrix3d(const std::string& name, const std::vector<double>& value) const {
        require_size(value.size(), 9, name);
        return nanousd_set_attribm3d(require(), name.c_str(), value.data()) != 0;
    }

    bool set_matrix4d(const std::string& name, const std::vector<double>& value) const {
        require_size(value.size(), 16, name);
        return nanousd_set_attribm4d(require(), name.c_str(), value.data()) != 0;
    }

    bool set_float_array(const std::string& name, const std::vector<float>& value) const {
        return nanousd_set_attribarrayf(require(), name.c_str(), data_or_null(value),
                                        static_cast<int>(value.size())) != 0;
    }

    bool set_double_array(const std::string& name, const std::vector<double>& value) const {
        return nanousd_set_attribarrayd(require(), name.c_str(), data_or_null(value),
                                        static_cast<int>(value.size())) != 0;
    }

    bool set_int_array(const std::string& name, const std::vector<int>& value) const {
        return nanousd_set_attribarrayi(require(), name.c_str(), data_or_null(value),
                                        static_cast<int>(value.size())) != 0;
    }

    bool set_vec3f_array(const std::string& name, const std::vector<float>& value) const {
        if (value.size() % 3 != 0) {
            throw std::runtime_error(name + " expects a flat vec3f array");
        }
        return nanousd_set_attribarrayv3f(require(), name.c_str(), data_or_null(value),
                                          static_cast<int>(value.size() / 3)) != 0;
    }

    bool set_vec3d_array(const std::string& name, const std::vector<double>& value) const {
        if (value.size() % 3 != 0) {
            throw std::runtime_error(name + " expects a flat vec3d array");
        }
        return nanousd_set_attribarrayv3d(require(), name.c_str(), data_or_null(value),
                                          static_cast<int>(value.size() / 3)) != 0;
    }

    bool set_string_array(const std::string& name, const std::vector<std::string>& value) const {
        std::vector<const char*> ptrs = string_ptrs(value);
        return nanousd_set_attribarrays(require(), name.c_str(), ptrs.data(),
                                        static_cast<int>(ptrs.size())) != 0;
    }

    bool set_token_array(const std::string& name, const std::vector<std::string>& value) const {
        std::vector<const char*> ptrs = string_ptrs(value);
        return nanousd_set_attribarraytokens(require(), name.c_str(), ptrs.data(),
                                             static_cast<int>(ptrs.size())) != 0;
    }

    bool set_attribute(const std::string& name, nb::handle value, const std::string& type_name = "") const {
        std::string type = type_name.empty() ? attribute_type(name) : type_name;
        if (type == "token") {
            return set_token(name, nb::cast<std::string>(value));
        }
        if (type == "asset") {
            return set_asset(name, nb::cast<std::string>(value));
        }
        if (type == "string") {
            return set_string(name, nb::cast<std::string>(value));
        }
        if (type == "bool") {
            return set_bool(name, nb::cast<bool>(value));
        }
        if (type == "int") {
            return set_int(name, nb::cast<int>(value));
        }
        if (type == "int64") {
            return set_int64(name, nb::cast<int64_t>(value));
        }
        if (type == "float") {
            return set_float(name, nb::cast<float>(value));
        }
        if (type == "double") {
            return set_double(name, nb::cast<double>(value));
        }
        if (type == "float[]") {
            return set_float_array(name, nb::cast<std::vector<float>>(value));
        }
        if (type == "double[]") {
            return set_double_array(name, nb::cast<std::vector<double>>(value));
        }
        if (type == "int[]") {
            return set_int_array(name, nb::cast<std::vector<int>>(value));
        }
        if (type == "string[]" || type == "asset[]") {
            return set_string_array(name, nb::cast<std::vector<std::string>>(value));
        }
        if (type == "token[]") {
            return set_token_array(name, nb::cast<std::vector<std::string>>(value));
        }
        if (contains(type, "3f[]")) {
            return set_vec3f_array(name, nb::cast<std::vector<float>>(value));
        }
        if (contains(type, "3d[]")) {
            return set_vec3d_array(name, nb::cast<std::vector<double>>(value));
        }
        if (contains(type, "2f")) {
            return set_vec2f(name, nb::cast<std::vector<float>>(value));
        }
        if (contains(type, "3f")) {
            return set_vec3f(name, nb::cast<std::vector<float>>(value));
        }
        if (contains(type, "4f")) {
            return set_vec4f(name, nb::cast<std::vector<float>>(value));
        }
        if (contains(type, "2d")) {
            return set_vec2d(name, nb::cast<std::vector<double>>(value));
        }
        if (contains(type, "3d")) {
            return set_vec3d(name, nb::cast<std::vector<double>>(value));
        }
        if (contains(type, "4d")) {
            return set_vec4d(name, nb::cast<std::vector<double>>(value));
        }
        if (contains(type, "2i")) {
            return set_vec2i(name, nb::cast<std::vector<int>>(value));
        }
        if (contains(type, "3i")) {
            return set_vec3i(name, nb::cast<std::vector<int>>(value));
        }
        if (contains(type, "4i")) {
            return set_vec4i(name, nb::cast<std::vector<int>>(value));
        }
        if (type == "quatf") {
            return set_quatf(name, nb::cast<std::vector<float>>(value));
        }
        if (type == "quatd") {
            return set_quatd(name, nb::cast<std::vector<double>>(value));
        }
        if (type == "matrix3d") {
            return set_matrix3d(name, nb::cast<std::vector<double>>(value));
        }
        if (type == "matrix4d") {
            return set_matrix4d(name, nb::cast<std::vector<double>>(value));
        }
        return false;
    }

    bool has_samples(const std::string& name) const {
        return nanousd_hassamples(require(), name.c_str()) != 0;
    }

    std::vector<double> sample_keys(const std::string& name) const {
        NanousdPrim p = require();
        std::vector<double> out;
        int n = nanousd_nsamplekeys(p, name.c_str());
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            out.push_back(nanousd_samplekey(p, name.c_str(), i));
        }
        return out;
    }

    std::optional<float> read_sample_float(const std::string& name, double time) const {
        int ok = 0;
        float value = nanousd_samplef(require(), name.c_str(), time, &ok);
        if (!ok) {
            return std::nullopt;
        }
        return value;
    }

    std::optional<double> read_sample_double(const std::string& name, double time) const {
        int ok = 0;
        double value = nanousd_sampled(require(), name.c_str(), time, &ok);
        if (!ok) {
            return std::nullopt;
        }
        return value;
    }

    std::optional<std::vector<float>> read_sample_vec2f(const std::string& name, double time) const {
        float v[2] = {};
        if (!nanousd_samplev2f(require(), name.c_str(), time, v)) {
            return std::nullopt;
        }
        return std::vector<float>{v[0], v[1]};
    }

    std::optional<std::vector<float>> read_sample_vec3f(const std::string& name, double time) const {
        float v[3] = {};
        if (!nanousd_samplev3f(require(), name.c_str(), time, v)) {
            return std::nullopt;
        }
        return std::vector<float>{v[0], v[1], v[2]};
    }

    std::optional<std::vector<double>> read_sample_vec3d(const std::string& name, double time) const {
        double v[3] = {};
        if (!nanousd_samplev3d(require(), name.c_str(), time, v)) {
            return std::nullopt;
        }
        return std::vector<double>{v[0], v[1], v[2]};
    }

    std::vector<float> read_sample_float_array(const std::string& name, double time, int maxlen) const {
        std::vector<float> out(static_cast<size_t>(maxlen > 0 ? maxlen : 0));
        int written = nanousd_samplearrayf(require(), name.c_str(), time, out.data(), maxlen);
        if (written < 0) {
            return {};
        }
        out.resize(static_cast<size_t>(written));
        return out;
    }

    std::vector<double> read_sample_double_array(const std::string& name, double time, int maxlen) const {
        std::vector<double> out(static_cast<size_t>(maxlen > 0 ? maxlen : 0));
        int written = nanousd_samplearrayd(require(), name.c_str(), time, out.data(), maxlen);
        if (written < 0) {
            return {};
        }
        out.resize(static_cast<size_t>(written));
        return out;
    }

    std::vector<int> read_sample_int_array(const std::string& name, double time, int maxlen) const {
        std::vector<int> out(static_cast<size_t>(maxlen > 0 ? maxlen : 0));
        int written = nanousd_samplearrayi(require(), name.c_str(), time, out.data(), maxlen);
        if (written < 0) {
            return {};
        }
        out.resize(static_cast<size_t>(written));
        return out;
    }

    std::vector<float> read_sample_float_array_flat(const std::string& name, double time, int components = 1) const {
        int n = nanousd_attribarraylen(require(), name.c_str());
        int maxlen = (n > 0 ? n : 65536) * (components > 0 ? components : 1);
        return read_sample_float_array(name, time, maxlen);
    }

    std::vector<double> read_sample_double_array_flat(const std::string& name, double time, int components = 1) const {
        int n = nanousd_attribarraylen(require(), name.c_str());
        int maxlen = (n > 0 ? n : 65536) * (components > 0 ? components : 1);
        return read_sample_double_array(name, time, maxlen);
    }

    std::vector<int> read_sample_int_array_flat(const std::string& name, double time, int components = 1) const {
        int n = nanousd_attribarraylen(require(), name.c_str());
        int maxlen = (n > 0 ? n : 65536) * (components > 0 ? components : 1);
        return read_sample_int_array(name, time, maxlen);
    }

    bool set_sample_float(const std::string& name, double time, float value) const {
        return nanousd_set_samplef(require(), name.c_str(), time, value) != 0;
    }

    bool set_sample_double(const std::string& name, double time, double value) const {
        return nanousd_set_sampled(require(), name.c_str(), time, value) != 0;
    }

    bool set_sample_vec3f(const std::string& name, double time, const std::vector<float>& value) const {
        require_size(value.size(), 3, name);
        return nanousd_set_samplev3f(require(), name.c_str(), time, value.data()) != 0;
    }

    bool set_sample_vec3d(const std::string& name, double time, const std::vector<double>& value) const {
        require_size(value.size(), 3, name);
        return nanousd_set_samplev3d(require(), name.c_str(), time, value.data()) != 0;
    }

    bool set_sample_vec4f(const std::string& name, double time, const std::vector<float>& value) const {
        require_size(value.size(), 4, name);
        return nanousd_set_samplev4f(require(), name.c_str(), time, value.data()) != 0;
    }

    bool set_sample_quatf(const std::string& name, double time, const std::vector<float>& value) const {
        require_size(value.size(), 4, name);
        return nanousd_set_sampleqf(require(), name.c_str(), time, value.data()) != 0;
    }

    bool set_sample_token(const std::string& name, double time, const std::string& value) const {
        return nanousd_set_sample_token(require(), name.c_str(), time, value.c_str()) != 0;
    }

    bool set_sample_vec2d(const std::string& name, double time, const std::vector<double>& value) const {
        require_size(value.size(), 2, name);
        return nanousd_set_samplev2d(require(), name.c_str(), time, value.data()) != 0;
    }

    bool set_sample_vec4d(const std::string& name, double time, const std::vector<double>& value) const {
        require_size(value.size(), 4, name);
        return nanousd_set_samplev4d(require(), name.c_str(), time, value.data()) != 0;
    }

    bool set_sample_matrix4d(const std::string& name, double time, const std::vector<double>& value) const {
        require_size(value.size(), 16, name);
        return nanousd_set_samplem4d(require(), name.c_str(), time, value.data()) != 0;
    }

    bool set_sample_float_array(const std::string& name, double time, const std::vector<float>& value) const {
        return nanousd_set_samplearrayf(require(), name.c_str(), time, data_or_null(value),
                                        static_cast<int>(value.size())) != 0;
    }

    bool set_sample_double_array(const std::string& name, double time, const std::vector<double>& value) const {
        return nanousd_set_samplearrayd(require(), name.c_str(), time, data_or_null(value),
                                        static_cast<int>(value.size())) != 0;
    }

    bool set_sample_int_array(const std::string& name, double time, const std::vector<int>& value) const {
        return nanousd_set_samplearrayi(require(), name.c_str(), time, data_or_null(value),
                                        static_cast<int>(value.size())) != 0;
    }

    bool set_sample_vec3f_array(const std::string& name, double time, const std::vector<float>& value) const {
        if (value.size() % 3 != 0) {
            throw std::runtime_error(name + " expects a flat vec3f array");
        }
        return nanousd_set_samplearrayv3f(require(), name.c_str(), time, data_or_null(value),
                                          static_cast<int>(value.size() / 3)) != 0;
    }

    bool set_sample_vec3d_array(const std::string& name, double time, const std::vector<double>& value) const {
        if (value.size() % 3 != 0) {
            throw std::runtime_error(name + " expects a flat vec3d array");
        }
        return nanousd_set_samplearrayv3d(require(), name.c_str(), time, data_or_null(value),
                                          static_cast<int>(value.size() / 3)) != 0;
    }

    bool clear_default(const std::string& name) const {
        return nanousd_clear_default(require(), name.c_str()) != 0;
    }

    bool clear_samples(const std::string& name) const {
        return nanousd_clear_samples(require(), name.c_str()) != 0;
    }

    bool block_attribute(const std::string& name) const {
        return nanousd_block_attrib(require(), name.c_str()) != 0;
    }

    bool has_relationship(const std::string& name) const {
        return nanousd_hasrel(require(), name.c_str()) != 0;
    }

    std::vector<std::string> relationship_names() const {
        NanousdPrim p = require();
        std::vector<std::string> out;
        std::unordered_set<std::string> seen;

        int n = nanousd_nattribs(p);
        for (int i = 0; i < n; ++i) {
            const char* raw_name = nanousd_attribname(p, i);
            if (!raw_name) {
                continue;
            }
            std::string name(raw_name);
            if (!is_known_relationship_name(name) && nanousd_hasrel(p, raw_name) == 0) {
                continue;
            }
            if (seen.insert(name).second) {
                out.push_back(std::move(name));
            }
        }

        static constexpr const char* candidates[] = {
            "material:binding",
            "material:binding:preview",
            "material:binding:full",
            "material:binding:physics",
            "outputs:surface.connect",
            "outputs:mdl:surface.connect",
        };
        for (const char* candidate : candidates) {
            if (seen.find(candidate) == seen.end() && nanousd_hasrel(p, candidate) != 0) {
                seen.insert(candidate);
                out.emplace_back(candidate);
            }
        }
        return out;
    }

    bool create_relationship(const std::string& name) const {
        return nanousd_create_rel(require(), name.c_str()) != 0;
    }

    std::vector<std::string> relationship_targets(const std::string& name) const {
        NanousdPrim p = require();
        std::vector<std::string> out;
        int n = nanousd_nreltargets(p, name.c_str());
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            out.push_back(cstr(nanousd_reltarget(p, name.c_str(), i)));
        }
        return out;
    }

    bool set_relationship_targets(const std::string& name, const std::vector<std::string>& targets) const {
        std::vector<const char*> ptrs = string_ptrs(targets);
        return nanousd_set_reltargets(require(), name.c_str(), ptrs.data(),
                                      static_cast<int>(ptrs.size())) != 0;
    }

    bool add_relationship_target(const std::string& name, const std::string& target) const {
        return nanousd_add_reltarget(require(), name.c_str(), target.c_str()) != 0;
    }

    bool clear_relationship_targets(const std::string& name) const {
        return nanousd_clear_reltargets(require(), name.c_str()) != 0;
    }

    bool has_connections(const std::string& name) const {
        return nanousd_hasconnections(require(), name.c_str()) != 0;
    }

    std::vector<std::string> connections(const std::string& name) const {
        NanousdPrim p = require();
        std::vector<std::string> out;
        int n = nanousd_nconnections(p, name.c_str());
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            out.push_back(cstr(nanousd_connection(p, name.c_str(), i)));
        }
        return out;
    }

    bool add_reference(const std::string& asset_path = "", const std::string& prim_path = "") const {
        return nanousd_add_reference(require(),
                                     asset_path.empty() ? nullptr : asset_path.c_str(),
                                     prim_path.empty() ? nullptr : prim_path.c_str()) != 0;
    }

    bool add_payload(const std::string& asset_path = "", const std::string& prim_path = "") const {
        return nanousd_add_payload(require(),
                                   asset_path.empty() ? nullptr : asset_path.c_str(),
                                   prim_path.empty() ? nullptr : prim_path.c_str()) != 0;
    }

    bool add_inherit(const std::string& prim_path) const {
        return nanousd_add_inherit(require(), prim_path.c_str()) != 0;
    }

    bool add_specialize(const std::string& prim_path) const {
        return nanousd_add_specialize(require(), prim_path.c_str()) != 0;
    }

    bool remove_listop_item(const std::string& field, int list_op_kind, int index) const {
        return nanousd_remove_listop_item(require(), field.c_str(), list_op_kind, index) != 0;
    }

    bool remove_reference(int index) const { return nanousd_remove_reference(require(), index) != 0; }
    bool remove_payload(int index) const { return nanousd_remove_payload(require(), index) != 0; }
    bool remove_inherit(int index) const { return nanousd_remove_inherit(require(), index) != 0; }
    bool remove_specialize(int index) const { return nanousd_remove_specialize(require(), index) != 0; }

    std::optional<ListOp> listop(const std::string& field) const {
        NanousdListOp op = nanousd_prim_listop(require(), field.c_str());
        if (!op) {
            return std::nullopt;
        }
        return ListOp(op);
    }

    std::vector<std::string> variant_set_names() const {
        NanousdPrim p = require();
        std::vector<std::string> out;
        int n = nanousd_nvariantsets(p);
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            out.push_back(cstr(nanousd_variantsetname(p, i)));
        }
        return out;
    }

    bool has_variant_set(const std::string& set_name) const {
        return nanousd_hasvariantset(require(), set_name.c_str()) != 0;
    }

    std::vector<std::string> variant_names(const std::string& set_name) const {
        NanousdPrim p = require();
        std::vector<std::string> out;
        int n = nanousd_nvariants(p, set_name.c_str());
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            out.push_back(cstr(nanousd_variantname(p, set_name.c_str(), i)));
        }
        return out;
    }

    std::string variant_selection(const std::string& set_name) const {
        return cstr(nanousd_variantselection(require(), set_name.c_str()));
    }

    bool set_variant_selection(const std::string& set_name,
                               const std::string& variant_name = "",
                               int layer_index = 0) const {
        return nanousd_setvariantselection(require(), set_name.c_str(),
                                           variant_name.empty() ? nullptr : variant_name.c_str(),
                                           layer_index) != 0;
    }

    bool create_variant_set(const std::string& set_name) const {
        return nanousd_create_variantset(require(), set_name.c_str()) != 0;
    }

    bool create_variant(const std::string& set_name, const std::string& variant_name) const {
        return nanousd_create_variant(require(), set_name.c_str(), variant_name.c_str()) != 0;
    }

    std::optional<std::string> metadata_string(const std::string& key) const {
        int ok = 0;
        const char* value = nanousd_prim_metadatas(require(), key.c_str(), &ok);
        if (!ok || !value) {
            return std::nullopt;
        }
        return std::string(value);
    }

    std::optional<double> metadata_double(const std::string& key) const {
        int ok = 0;
        double value = nanousd_prim_metadatad(require(), key.c_str(), &ok);
        if (!ok) {
            return std::nullopt;
        }
        return value;
    }

    bool set_metadata_string(const std::string& key, const std::string& value) const {
        return nanousd_set_prim_metadatas(require(), key.c_str(), value.c_str()) != 0;
    }

    bool set_metadata_double(const std::string& key, double value) const {
        return nanousd_set_prim_metadatad(require(), key.c_str(), value) != 0;
    }

    bool set_metadata_token(const std::string& key, const std::string& value) const {
        return nanousd_set_prim_metadata_token(require(), key.c_str(), value.c_str()) != 0;
    }

private:
    std::shared_ptr<PrimHandle> handle_;

    NanousdPrim require() const {
        if (!handle_ || !handle_->prim) {
            throw std::runtime_error("invalid nanousd Prim");
        }
        return handle_->prim;
    }

    Prim make_prim(NanousdPrim prim) const {
        return Prim(std::make_shared<PrimHandle>(handle_->stage, prim));
    }

    template <typename T, typename ReadFn>
    std::vector<T> read_numeric_array(const std::string& name, ReadFn read_fn) const {
        NanousdPrim p = require();
        int n = nanousd_attribarraylen(p, name.c_str());
        if (n <= 0) {
            return {};
        }
        std::vector<T> out(static_cast<size_t>(n));
        int written = read_fn(p, name.c_str(), out.data(), n);
        if (written < 0) {
            return {};
        }
        out.resize(static_cast<size_t>(written));
        return out;
    }

    template <typename T, typename ReadFn>
    std::vector<T> read_numeric_array_flat(const std::string& name, ReadFn read_fn, int components) const {
        NanousdPrim p = require();
        int n = nanousd_attribarraylen(p, name.c_str());
        if (n <= 0) {
            return {};
        }
        int flat_n = n * (components > 0 ? components : 1);
        std::vector<T> out(static_cast<size_t>(flat_n));
        int written = read_fn(p, name.c_str(), out.data(), flat_n);
        if (written < 0) {
            return {};
        }
        out.resize(static_cast<size_t>(written));
        return out;
    }

    template <typename T, typename ReadFn>
    nb::object read_numeric_array_numpy(const std::string& name, ReadFn read_fn, int components) const {
        NanousdPrim p = require();
        int n = nanousd_attribarraylen(p, name.c_str());
        if (n < 0) {
            return nb::none();
        }
        int cols = components > 0 ? components : 1;
        int flat_n = n * cols;
        std::vector<T> out(static_cast<size_t>(flat_n));
        int written = flat_n > 0 ? read_fn(p, name.c_str(), out.data(), flat_n) : 0;
        if (written < 0 || (cols > 1 && written % cols != 0)) {
            return nb::none();
        }
        out.resize(static_cast<size_t>(written));
        return vector_to_numpy(std::move(out), cols);
    }
};

class Stage {
public:
    Stage() = default;
    explicit Stage(std::shared_ptr<StageState> state) : state_(std::move(state)) {}

    static Stage open(const std::string& filepath) {
        auto state = std::make_shared<StageState>();
        state->stage = nanousd_open(filepath.c_str());
        if (!state->stage) {
            throw std::runtime_error("nanousd_open returned null");
        }
        if (!nanousd_isvalid(state->stage)) {
            std::string err = cstr(nanousd_error(state->stage));
            throw std::runtime_error(err.empty() ? "invalid nanousd stage" : err);
        }
        return Stage(std::move(state));
    }

    static Stage create() {
        auto state = std::make_shared<StageState>();
        state->stage = nanousd_create();
        if (!state->stage) {
            throw std::runtime_error("nanousd_create returned null");
        }
        return Stage(std::move(state));
    }

    bool valid() const {
        return state_ && state_->stage && nanousd_isvalid(state_->stage);
    }

    std::string error() const {
        return state_ && state_->stage ? cstr(nanousd_error(state_->stage)) : "invalid nanousd Stage";
    }

    std::string root_layer_path() const {
        return cstr(nanousd_stage_get_root_layer_path(require()));
    }

    std::vector<std::string> used_layers() const {
        NanousdStage s = require();
        std::vector<std::string> out;
        int n = nanousd_stage_n_layers(s);
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            out.push_back(cstr(nanousd_stage_layer_path(s, i)));
        }
        return out;
    }

    int num_layers() const { return nanousd_stage_n_layers(require()); }

    std::string layer_path(int layer_index) const {
        return cstr(nanousd_stage_layer_path(require(), layer_index));
    }

    bool layer_has_prim_spec(int layer_index, const std::string& prim_path) const {
        return nanousd_layer_has_prim_spec(require(), layer_index, prim_path.c_str()) != 0;
    }

    bool layer_has_attr_opinion(int layer_index,
                                const std::string& prim_path,
                                const std::string& attr_name) const {
        return nanousd_layer_has_attr_opinion(require(), layer_index,
                                              prim_path.c_str(), attr_name.c_str()) != 0;
    }

    int layer_attr_num_samples(int layer_index,
                               const std::string& prim_path,
                               const std::string& attr_name) const {
        return nanousd_layer_attr_nsamples(require(), layer_index,
                                           prim_path.c_str(), attr_name.c_str());
    }

    std::optional<ListOp> layer_prim_listop(int layer_index,
                                            const std::string& prim_path,
                                            const std::string& field) const {
        NanousdListOp op = nanousd_layer_prim_listop(require(), layer_index,
                                                     prim_path.c_str(), field.c_str());
        if (!op) {
            return std::nullopt;
        }
        return ListOp(op);
    }

    std::vector<std::string> layer_sublayer_paths(int layer_index) const {
        NanousdStage s = require();
        std::vector<std::string> out;
        int n = nanousd_layer_n_sublayers(s, layer_index);
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            out.push_back(cstr(nanousd_layer_sublayer_path(s, layer_index, i)));
        }
        return out;
    }

    std::optional<std::vector<double>> layer_offset(int layer_index) const {
        double offset = 0.0;
        double scale = 1.0;
        if (!nanousd_layer_offset(require(), layer_index, &offset, &scale)) {
            return std::nullopt;
        }
        return std::vector<double>{offset, scale};
    }

    std::vector<Prim> traverse() const {
        NanousdStage s = require();
        std::vector<Prim> out;
        int n = nanousd_nprims(s);
        out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
        for (int i = 0; i < n; ++i) {
            NanousdPrim prim = nanousd_prim(s, i);
            if (prim) {
                out.emplace_back(make_prim(prim));
            }
        }
        return out;
    }

    std::optional<Prim> get_prim_at_path(const std::string& path) const {
        NanousdPrim prim = nanousd_primpath(require(), path.c_str());
        if (!prim) {
            return std::nullopt;
        }
        return make_prim(prim);
    }

    std::optional<Prim> default_prim() const {
        NanousdPrim prim = nanousd_defaultprim(require());
        if (!prim) {
            return std::nullopt;
        }
        return make_prim(prim);
    }

    Prim define_prim(const std::string& path,
                     const std::string& type_name = "",
                     const std::string& specifier = "def") const {
        NanousdPrim prim = nullptr;
        if (specifier.empty() || specifier == "def") {
            prim = nanousd_define_prim(require(), path.c_str(),
                                       type_name.empty() ? nullptr : type_name.c_str());
        } else {
            prim = nanousd_define_prim_s(require(), path.c_str(),
                                         type_name.empty() ? nullptr : type_name.c_str(),
                                         specifier.c_str());
        }
        if (!prim) {
            throw std::runtime_error("nanousd_define_prim failed");
        }
        return make_prim(prim);
    }

    double time_codes_per_second() const { return nanousd_timecodes_per_second(require()); }
    double frames_per_second() const { return nanousd_frames_per_second(require()); }
    double start_time() const { return nanousd_start_time(require()); }
    double end_time() const { return nanousd_end_time(require()); }

    std::optional<double> metadata_double(const std::string& key) const {
        int ok = 0;
        double value = nanousd_metadatad(require(), key.c_str(), &ok);
        if (!ok) {
            return std::nullopt;
        }
        return value;
    }

    std::optional<std::string> metadata_string(const std::string& key) const {
        int ok = 0;
        const char* value = nanousd_metadatas(require(), key.c_str(), &ok);
        if (!ok || !value) {
            if (key != "upAxis" && key != "defaultPrim" && key != "documentation" && key != "comment") {
                return std::nullopt;
            }
            return find_usda_layer_metadata_string(to_usda_string(), key);
        }
        return std::string(value);
    }

    bool set_metadata_double(const std::string& key, double value) const {
        return nanousd_set_stage_metadatad(require(), key.c_str(), value) != 0;
    }

    bool set_metadata_string(const std::string& key, const std::string& value) const {
        return nanousd_set_stage_metadatas(require(), key.c_str(), value.c_str()) != 0;
    }

    bool set_metadata_token(const std::string& key, const std::string& value) const {
        return nanousd_set_stage_metadata_token(require(), key.c_str(), value.c_str()) != 0;
    }

    bool write_usda(const std::string& filepath) const {
        return nanousd_write_usda(require(), filepath.c_str()) != 0;
    }

    bool write_usdc(const std::string& filepath) const {
        return nanousd_write_usdc(require(), filepath.c_str()) != 0;
    }

    std::string to_usda_string() const {
        return owned_string(nanousd_write_usda_string(require()));
    }

    std::vector<Diagnostic> diagnostics() const {
        int count = 0;
        NanousdDiagnostic* raw = nanousd_diagnostics(require(), &count);
        std::vector<Diagnostic> out;
        out.reserve(count > 0 ? static_cast<size_t>(count) : 0);
        for (int i = 0; i < count; ++i) {
            Diagnostic diag;
            diag.severity = raw[i].severity;
            diag.category = raw[i].category;
            diag.message = cstr(raw[i].message);
            diag.prim_path = cstr(raw[i].prim_path);
            diag.layer_path = cstr(raw[i].layer_path);
            diag.asset_path = cstr(raw[i].asset_path);
            diag.arc_type = raw[i].arc_type;
            out.push_back(std::move(diag));
        }
        nanousd_free_diagnostics(raw, count);
        return out;
    }

    std::string diagnostics_json() const {
        return owned_string(nanousd_diagnostics_json(require()));
    }

private:
    std::shared_ptr<StageState> state_;

    NanousdStage require() const {
        if (!state_ || !state_->stage) {
            throw std::runtime_error("invalid nanousd Stage");
        }
        return state_->stage;
    }

    Prim make_prim(NanousdPrim prim) const {
        return Prim(std::make_shared<PrimHandle>(state_, prim));
    }
};

static std::vector<float> checked_vec3f(const std::vector<float>& value, const std::string& name) {
    require_size(value.size(), 3, name);
    return value;
}

static std::vector<double> checked_vec3d(const std::vector<double>& value, const std::string& name) {
    require_size(value.size(), 3, name);
    return value;
}

static std::vector<double> checked_matrix4d(const std::vector<double>& value, const std::string& name) {
    require_size(value.size(), 16, name);
    return value;
}

NB_MODULE(_nanousd, m) {
    m.doc() = "nanobind wrapper over the nanousd C API";

    m.def("sdf_path_is_valid", &sdf_path_is_valid_impl, "text"_a);
    m.def("sdf_path_validated_text", &sdf_path_validated_text, "text"_a);
    m.def("fan_triangulate_indices", &fan_triangulate_indices_numpy,
          "counts"_a, "indices"_a, "flip_winding"_a = false);

    nb::class_<Diagnostic>(m, "Diagnostic")
        .def_ro("severity", &Diagnostic::severity)
        .def_ro("category", &Diagnostic::category)
        .def_ro("message", &Diagnostic::message)
        .def_ro("prim_path", &Diagnostic::prim_path)
        .def_ro("layer_path", &Diagnostic::layer_path)
        .def_ro("asset_path", &Diagnostic::asset_path)
        .def_ro("arc_type", &Diagnostic::arc_type);

    nb::class_<Path>(m, "Path")
        .def(nb::init<const std::string&>(), "text"_a)
        .def_static("parse", &Path::parse, "text"_a)
        .def_prop_ro("is_valid", &Path::valid)
        .def_prop_ro("name", &Path::name)
        .def_prop_ro("is_absolute", &Path::is_absolute)
        .def_prop_ro("is_root", &Path::is_root)
        .def_prop_ro("is_property", &Path::is_property)
        .def("append_child", &Path::append_child, "child"_a)
        .def("append_property", &Path::append_property, "property"_a)
        .def("parent", &Path::parent)
        .def("equal", &Path::equal, "other"_a)
        .def("__str__", &Path::str)
        .def("__repr__", [](const Path& path) { return "Path('" + path.str() + "')"; })
        .def("__eq__", [](const Path& a, const Path& b) { return a.equal(b); });

    nb::class_<ListOp>(m, "ListOp")
        .def_static("create_explicit", &ListOp::create_explicit, "items"_a)
        .def_static("create", &ListOp::create,
                    "prepended"_a = std::vector<std::string>{},
                    "appended"_a = std::vector<std::string>{},
                    "deleted"_a = std::vector<std::string>{})
        .def_static("combine", &ListOp::combine, "stronger"_a, "weaker"_a)
        .def_prop_ro("is_valid", &ListOp::valid)
        .def_prop_ro("is_explicit", &ListOp::is_explicit)
        .def_prop_ro("items", &ListOp::items)
        .def_prop_ro("prepended_items", &ListOp::prepended_items)
        .def_prop_ro("appended_items", &ListOp::appended_items)
        .def_prop_ro("deleted_items", &ListOp::deleted_items);

    nb::class_<Stage>(m, "Stage")
        .def_static("open", &Stage::open, "filepath"_a)
        .def_static("create", &Stage::create)
        .def_prop_ro("is_valid", &Stage::valid)
        .def_prop_ro("error", &Stage::error)
        .def_prop_ro("root_layer_path", &Stage::root_layer_path)
        .def_prop_ro("used_layers", &Stage::used_layers)
        .def_prop_ro("num_layers", &Stage::num_layers)
        .def_prop_ro("time_codes_per_second", &Stage::time_codes_per_second)
        .def_prop_ro("frames_per_second", &Stage::frames_per_second)
        .def_prop_ro("start_time", &Stage::start_time)
        .def_prop_ro("end_time", &Stage::end_time)
        .def("layer_path", &Stage::layer_path, "layer_index"_a)
        .def("layer_has_prim_spec", &Stage::layer_has_prim_spec, "layer_index"_a, "prim_path"_a)
        .def("layer_has_attr_opinion", &Stage::layer_has_attr_opinion,
             "layer_index"_a, "prim_path"_a, "attr_name"_a)
        .def("layer_attr_num_samples", &Stage::layer_attr_num_samples,
             "layer_index"_a, "prim_path"_a, "attr_name"_a)
        .def("layer_prim_listop", &Stage::layer_prim_listop,
             "layer_index"_a, "prim_path"_a, "field"_a)
        .def("layer_sublayer_paths", &Stage::layer_sublayer_paths, "layer_index"_a)
        .def("layer_offset", &Stage::layer_offset, "layer_index"_a)
        .def("traverse", &Stage::traverse)
        .def("get_prim_at_path", &Stage::get_prim_at_path, "path"_a)
        .def("default_prim", &Stage::default_prim)
        .def("define_prim", &Stage::define_prim,
             "path"_a, "type_name"_a = "", "specifier"_a = "def")
        .def("metadata_double", &Stage::metadata_double, "key"_a)
        .def("metadata_string", &Stage::metadata_string, "key"_a)
        .def("set_metadata_double", &Stage::set_metadata_double, "key"_a, "value"_a)
        .def("set_metadata_string", &Stage::set_metadata_string, "key"_a, "value"_a)
        .def("set_metadata_token", &Stage::set_metadata_token, "key"_a, "value"_a)
        .def("write_usda", &Stage::write_usda, "filepath"_a)
        .def("write_usdc", &Stage::write_usdc, "filepath"_a)
        .def("export", &Stage::write_usda, "filepath"_a)
        .def("to_usda_string", &Stage::to_usda_string)
        .def("to_string", &Stage::to_usda_string)
        .def("diagnostics", &Stage::diagnostics)
        .def("diagnostics_json", &Stage::diagnostics_json);

    nb::class_<Prim>(m, "Prim")
        .def_prop_ro("is_valid", &Prim::valid)
        .def_prop_ro("path", &Prim::path)
        .def_prop_ro("name", &Prim::name)
        .def_prop_ro("type_name", &Prim::type_name)
        .def_prop_ro("kind", &Prim::kind)
        .def_prop_ro("is_active", &Prim::is_active)
        .def_prop_ro("is_defined", &Prim::is_defined)
        .def_prop_ro("is_abstract", &Prim::is_abstract)
        .def_prop_ro("is_instanceable", &Prim::is_instanceable)
        .def_prop_ro("is_instance", &Prim::is_instance)
        .def_prop_ro("is_prototype", &Prim::is_prototype)
        .def_prop_ro("is_in_prototype", &Prim::is_in_prototype)
        .def("is_a", &Prim::is_a, "type_name"_a)
        .def("has_api", &Prim::has_api, "api_name"_a)
        .def("apply_api", &Prim::apply_api, "api_name"_a)
        .def("remove_api", &Prim::remove_api, "api_name"_a)
        .def("set_active", &Prim::set_active, "active"_a)
        .def("set_instanceable", &Prim::set_instanceable, "instanceable"_a)
        .def("set_specifier", &Prim::set_specifier, "specifier"_a)
        .def("remove", &Prim::remove)
        .def("children", &Prim::children)
        .def("child", &Prim::child, "name"_a)
        .def("parent", &Prim::parent)
        .def("prototype", &Prim::prototype)
        .def("instances", &Prim::instances)
        .def("attribute_names", &Prim::attribute_names)
        .def("attribute_names_in_namespace", &Prim::attribute_names_in_namespace, "namespace"_a)
        .def("authored_attribute_names", &Prim::authored_attribute_names)
        .def("authored_attribute_names_in_namespace",
             &Prim::authored_attribute_names_in_namespace, "namespace"_a)
        .def("authored_attribute_values_in_namespace",
             &Prim::authored_attribute_values_in_namespace, "namespace"_a)
        .def("authored_attribute_values_in_namespaces",
             &Prim::authored_attribute_values_in_namespaces, "namespaces"_a)
        .def("authored_attribute_values", &Prim::authored_attribute_values, "names"_a)
        .def("has_attribute", &Prim::has_attribute, "name"_a)
        .def("is_attribute_authored", &Prim::is_attribute_authored, "name"_a)
        .def("attribute_type", &Prim::attribute_type, "name"_a)
        .def("attribute_interpolation", &Prim::attribute_interpolation, "name"_a)
        .def("create_attribute", &Prim::create_attribute, "name"_a, "type_name"_a)
        .def("get", &Prim::get_attribute, "name"_a)
        .def("set", &Prim::set_attribute, "name"_a, "value"_a, "type_name"_a = "")
        .def("read_double", &Prim::read_double, "name"_a)
        .def("read_float", &Prim::read_float, "name"_a)
        .def("read_int", &Prim::read_int, "name"_a)
        .def("read_int64", &Prim::read_int64, "name"_a)
        .def("read_bool", &Prim::read_bool, "name"_a)
        .def("read_string", &Prim::read_string, "name"_a)
        .def("read_token", &Prim::read_token, "name"_a)
        .def("read_asset", &Prim::read_asset, "name"_a)
        .def("read_vec2f", &Prim::read_vec2f, "name"_a)
        .def("read_vec3f", &Prim::read_vec3f, "name"_a)
        .def("read_vec4f", &Prim::read_vec4f, "name"_a)
        .def("read_vec2d", &Prim::read_vec2d, "name"_a)
        .def("read_vec3d", &Prim::read_vec3d, "name"_a)
        .def("read_vec4d", &Prim::read_vec4d, "name"_a)
        .def("read_vec2i", &Prim::read_vec2i, "name"_a)
        .def("read_vec3i", &Prim::read_vec3i, "name"_a)
        .def("read_vec4i", &Prim::read_vec4i, "name"_a)
        .def("read_quatf", &Prim::read_quatf, "name"_a)
        .def("read_quatd", &Prim::read_quatd, "name"_a)
        .def("read_matrix3d", &Prim::read_matrix3d, "name"_a)
        .def("read_matrix4d", &Prim::read_matrix4d, "name"_a)
        .def("local_transform", &Prim::local_transform)
        .def("local_transform_at", &Prim::local_transform_at, "time"_a)
        .def("local_transform_info_at", &Prim::local_transform_info_at, "time"_a)
        .def("array_len", &Prim::array_len, "name"_a)
        .def("read_float_array", &Prim::read_float_array, "name"_a)
        .def("read_double_array", &Prim::read_double_array, "name"_a)
        .def("read_int_array", &Prim::read_int_array, "name"_a)
        .def("read_int64_array", &Prim::read_int64_array, "name"_a)
        .def("read_float_array_flat", &Prim::read_float_array_flat, "name"_a, "components"_a = 1)
        .def("read_double_array_flat", &Prim::read_double_array_flat, "name"_a, "components"_a = 1)
        .def("read_int_array_flat", &Prim::read_int_array_flat, "name"_a, "components"_a = 1)
        .def("read_float_array_numpy", &Prim::read_float_array_numpy, "name"_a, "components"_a = 1)
        .def("read_double_array_numpy", &Prim::read_double_array_numpy, "name"_a, "components"_a = 1)
        .def("read_int_array_numpy", &Prim::read_int_array_numpy, "name"_a, "components"_a = 1)
        .def("read_int64_array_numpy", &Prim::read_int64_array_numpy, "name"_a)
        .def("read_vec3f_array", &Prim::read_vec3f_array, "name"_a)
        .def("read_vec3d_array", &Prim::read_vec3d_array, "name"_a)
        .def("read_string_array", &Prim::read_string_array, "name"_a)
        .def("read_token_array", &Prim::read_token_array, "name"_a)
        .def("set_double", &Prim::set_double, "name"_a, "value"_a)
        .def("set_float", &Prim::set_float, "name"_a, "value"_a)
        .def("set_int", &Prim::set_int, "name"_a, "value"_a)
        .def("set_int64", &Prim::set_int64, "name"_a, "value"_a)
        .def("set_bool", &Prim::set_bool, "name"_a, "value"_a)
        .def("set_string", &Prim::set_string, "name"_a, "value"_a)
        .def("set_token", &Prim::set_token, "name"_a, "value"_a)
        .def("set_asset", &Prim::set_asset, "name"_a, "value"_a)
        .def("set_vec2f", &Prim::set_vec2f, "name"_a, "value"_a)
        .def("set_vec3f", &Prim::set_vec3f, "name"_a, "value"_a)
        .def("set_vec4f", &Prim::set_vec4f, "name"_a, "value"_a)
        .def("set_vec2d", &Prim::set_vec2d, "name"_a, "value"_a)
        .def("set_vec3d", &Prim::set_vec3d, "name"_a, "value"_a)
        .def("set_vec4d", &Prim::set_vec4d, "name"_a, "value"_a)
        .def("set_vec2i", &Prim::set_vec2i, "name"_a, "value"_a)
        .def("set_vec3i", &Prim::set_vec3i, "name"_a, "value"_a)
        .def("set_vec4i", &Prim::set_vec4i, "name"_a, "value"_a)
        .def("set_quatf", &Prim::set_quatf, "name"_a, "value"_a)
        .def("set_quatd", &Prim::set_quatd, "name"_a, "value"_a)
        .def("set_matrix3d", &Prim::set_matrix3d, "name"_a, "value"_a)
        .def("set_matrix4d", &Prim::set_matrix4d, "name"_a, "value"_a)
        .def("set_float_array", &Prim::set_float_array, "name"_a, "value"_a)
        .def("set_double_array", &Prim::set_double_array, "name"_a, "value"_a)
        .def("set_int_array", &Prim::set_int_array, "name"_a, "value"_a)
        .def("set_vec3f_array", &Prim::set_vec3f_array, "name"_a, "value"_a)
        .def("set_vec3d_array", &Prim::set_vec3d_array, "name"_a, "value"_a)
        .def("set_string_array", &Prim::set_string_array, "name"_a, "value"_a)
        .def("set_token_array", &Prim::set_token_array, "name"_a, "value"_a)
        .def("has_samples", &Prim::has_samples, "name"_a)
        .def("sample_keys", &Prim::sample_keys, "name"_a)
        .def("read_sample_float", &Prim::read_sample_float, "name"_a, "time"_a)
        .def("read_sample_double", &Prim::read_sample_double, "name"_a, "time"_a)
        .def("read_sample_vec2f", &Prim::read_sample_vec2f, "name"_a, "time"_a)
        .def("read_sample_vec3f", &Prim::read_sample_vec3f, "name"_a, "time"_a)
        .def("read_sample_vec3d", &Prim::read_sample_vec3d, "name"_a, "time"_a)
        .def("read_sample_float_array", &Prim::read_sample_float_array, "name"_a, "time"_a, "maxlen"_a)
        .def("read_sample_double_array", &Prim::read_sample_double_array, "name"_a, "time"_a, "maxlen"_a)
        .def("read_sample_int_array", &Prim::read_sample_int_array, "name"_a, "time"_a, "maxlen"_a)
        .def("read_sample_float_array_flat", &Prim::read_sample_float_array_flat,
             "name"_a, "time"_a, "components"_a = 1)
        .def("read_sample_double_array_flat", &Prim::read_sample_double_array_flat,
             "name"_a, "time"_a, "components"_a = 1)
        .def("read_sample_int_array_flat", &Prim::read_sample_int_array_flat,
             "name"_a, "time"_a, "components"_a = 1)
        .def("set_sample_float", &Prim::set_sample_float, "name"_a, "time"_a, "value"_a)
        .def("set_sample_double", &Prim::set_sample_double, "name"_a, "time"_a, "value"_a)
        .def("set_sample_vec3f", &Prim::set_sample_vec3f, "name"_a, "time"_a, "value"_a)
        .def("set_sample_vec3d", &Prim::set_sample_vec3d, "name"_a, "time"_a, "value"_a)
        .def("set_sample_vec4f", &Prim::set_sample_vec4f, "name"_a, "time"_a, "value"_a)
        .def("set_sample_quatf", &Prim::set_sample_quatf, "name"_a, "time"_a, "value"_a)
        .def("set_sample_token", &Prim::set_sample_token, "name"_a, "time"_a, "value"_a)
        .def("set_sample_vec2d", &Prim::set_sample_vec2d, "name"_a, "time"_a, "value"_a)
        .def("set_sample_vec4d", &Prim::set_sample_vec4d, "name"_a, "time"_a, "value"_a)
        .def("set_sample_matrix4d", &Prim::set_sample_matrix4d, "name"_a, "time"_a, "value"_a)
        .def("set_sample_float_array", &Prim::set_sample_float_array, "name"_a, "time"_a, "value"_a)
        .def("set_sample_double_array", &Prim::set_sample_double_array, "name"_a, "time"_a, "value"_a)
        .def("set_sample_int_array", &Prim::set_sample_int_array, "name"_a, "time"_a, "value"_a)
        .def("set_sample_vec3f_array", &Prim::set_sample_vec3f_array, "name"_a, "time"_a, "value"_a)
        .def("set_sample_vec3d_array", &Prim::set_sample_vec3d_array, "name"_a, "time"_a, "value"_a)
        .def("clear_default", &Prim::clear_default, "name"_a)
        .def("clear_samples", &Prim::clear_samples, "name"_a)
        .def("block_attribute", &Prim::block_attribute, "name"_a)
        .def("has_relationship", &Prim::has_relationship, "name"_a)
        .def("relationship_names", &Prim::relationship_names)
        .def("create_relationship", &Prim::create_relationship, "name"_a)
        .def("relationship_targets", &Prim::relationship_targets, "name"_a)
        .def("set_relationship_targets", &Prim::set_relationship_targets, "name"_a, "targets"_a)
        .def("add_relationship_target", &Prim::add_relationship_target, "name"_a, "target"_a)
        .def("clear_relationship_targets", &Prim::clear_relationship_targets, "name"_a)
        .def("has_connections", &Prim::has_connections, "name"_a)
        .def("connections", &Prim::connections, "name"_a)
        .def("add_reference", &Prim::add_reference, "asset_path"_a = "", "prim_path"_a = "")
        .def("add_payload", &Prim::add_payload, "asset_path"_a = "", "prim_path"_a = "")
        .def("add_inherit", &Prim::add_inherit, "prim_path"_a)
        .def("add_specialize", &Prim::add_specialize, "prim_path"_a)
        .def("remove_listop_item", &Prim::remove_listop_item, "field"_a, "list_op_kind"_a, "index"_a)
        .def("remove_reference", &Prim::remove_reference, "index"_a)
        .def("remove_payload", &Prim::remove_payload, "index"_a)
        .def("remove_inherit", &Prim::remove_inherit, "index"_a)
        .def("remove_specialize", &Prim::remove_specialize, "index"_a)
        .def("listop", &Prim::listop, "field"_a)
        .def("variant_set_names", &Prim::variant_set_names)
        .def("has_variant_set", &Prim::has_variant_set, "set_name"_a)
        .def("variant_names", &Prim::variant_names, "set_name"_a)
        .def("variant_selection", &Prim::variant_selection, "set_name"_a)
        .def("set_variant_selection", &Prim::set_variant_selection,
             "set_name"_a, "variant_name"_a = "", "layer_index"_a = 0)
        .def("create_variant_set", &Prim::create_variant_set, "set_name"_a)
        .def("create_variant", &Prim::create_variant, "set_name"_a, "variant_name"_a)
        .def("metadata_string", &Prim::metadata_string, "key"_a)
        .def("metadata_double", &Prim::metadata_double, "key"_a)
        .def("set_metadata_string", &Prim::set_metadata_string, "key"_a, "value"_a)
        .def("set_metadata_double", &Prim::set_metadata_double, "key"_a, "value"_a)
        .def("set_metadata_token", &Prim::set_metadata_token, "key"_a, "value"_a);

    m.def("register_schemas_json",
          [](const std::string& json) { return nanousd_register_schemas_json(json.c_str()) != 0; },
          "json"_a);

    m.def("dot3f", [](const std::vector<float>& a, const std::vector<float>& b) {
        std::vector<float> aa = checked_vec3f(a, "a");
        std::vector<float> bb = checked_vec3f(b, "b");
        return nanousd_dot3f(aa.data(), bb.data());
    }, "a"_a, "b"_a);

    m.def("dot3d", [](const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> aa = checked_vec3d(a, "a");
        std::vector<double> bb = checked_vec3d(b, "b");
        return nanousd_dot3d(aa.data(), bb.data());
    }, "a"_a, "b"_a);

    m.def("length3f", [](const std::vector<float>& v) {
        std::vector<float> vv = checked_vec3f(v, "v");
        return nanousd_length3f(vv.data());
    }, "v"_a);

    m.def("length3d", [](const std::vector<double>& v) {
        std::vector<double> vv = checked_vec3d(v, "v");
        return nanousd_length3d(vv.data());
    }, "v"_a);

    m.def("normalize3f", [](const std::vector<float>& v) {
        std::vector<float> vv = checked_vec3f(v, "v");
        float out[3] = {};
        nanousd_normalize3f(vv.data(), out);
        return std::vector<float>{out[0], out[1], out[2]};
    }, "v"_a);

    m.def("normalize3d", [](const std::vector<double>& v) {
        std::vector<double> vv = checked_vec3d(v, "v");
        double out[3] = {};
        nanousd_normalize3d(vv.data(), out);
        return std::vector<double>{out[0], out[1], out[2]};
    }, "v"_a);

    m.def("cross3f", [](const std::vector<float>& a, const std::vector<float>& b) {
        std::vector<float> aa = checked_vec3f(a, "a");
        std::vector<float> bb = checked_vec3f(b, "b");
        float out[3] = {};
        nanousd_cross3f(aa.data(), bb.data(), out);
        return std::vector<float>{out[0], out[1], out[2]};
    }, "a"_a, "b"_a);

    m.def("cross3d", [](const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> aa = checked_vec3d(a, "a");
        std::vector<double> bb = checked_vec3d(b, "b");
        double out[3] = {};
        nanousd_cross3d(aa.data(), bb.data(), out);
        return std::vector<double>{out[0], out[1], out[2]};
    }, "a"_a, "b"_a);

    m.def("mul_matrix4d", [](const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> aa = checked_matrix4d(a, "a");
        std::vector<double> bb = checked_matrix4d(b, "b");
        double out[16] = {};
        nanousd_mul_m4d(aa.data(), bb.data(), out);
        return std::vector<double>(out, out + 16);
    }, "a"_a, "b"_a);

    m.def("transform_point3d", [](const std::vector<double>& m4d, const std::vector<double>& point) {
        std::vector<double> mm = checked_matrix4d(m4d, "m4d");
        std::vector<double> pp = checked_vec3d(point, "point");
        double out[3] = {};
        nanousd_transform_point3d(mm.data(), pp.data(), out);
        return std::vector<double>{out[0], out[1], out[2]};
    }, "m4d"_a, "point"_a);

    m.def("quat_slerp", [](const std::vector<double>& a, const std::vector<double>& b, double t) {
        require_size(a.size(), 4, "a");
        require_size(b.size(), 4, "b");
        double out[4] = {};
        nanousd_quat_slerp(a.data(), b.data(), t, out);
        return std::vector<double>{out[0], out[1], out[2], out[3]};
    }, "a"_a, "b"_a, "t"_a);

    m.def("quat_to_matrix", [](const std::vector<double>& q) {
        require_size(q.size(), 4, "q");
        double out[16] = {};
        nanousd_quat_to_matrix(q.data(), out);
        return std::vector<double>(out, out + 16);
    }, "q"_a);
}
