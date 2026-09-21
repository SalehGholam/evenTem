/* Copyright (C) 2025 Thomas Friedrich, Chu-Ping Yu, Arno Annys
 * University of Antwerp - All Rights Reserved.
 * You may use, distribute and modify
 * this code under the terms of the GPL3 license.
 * You should have received a copy of the GPL3 license with
 * this file. If not, please visit:
 * https://www.gnu.org/licenses/gpl-3.0.en.html
 *
 * Authors:
 *   Thomas Friedrich <>
 *   Chu-Ping Yu <>
 *   Arno Annys <arno.annys@uantwerpen.be>
 */

#ifndef TPX3SCAN_SIDECAR_HPP
#define TPX3SCAN_SIDECAR_HPP

// Reads the ``.tpx3scan`` sidecar pyeventem (a separate, pure-Python/numba
// reimplementation of this project -- see ../../../../pyeventem) writes
// beside a ``.tpx3`` acquisition: the whole-file line-trigger scan
// (pyeventem's "pass 1"), persisted so it never has to be redone. Real
// interop, not a parallel format of our own -- this reads the EXACT file
// pyeventem's own ``python -m pyeventem.tools.build_index`` writes
// (pyeventem/src/pyeventem/decode/sidecar.py), so a folder indexed once
// (by either project) benefits both.
//
// **File format** (pyeventem's choice, documented in sidecar.py): an
// uncompressed ``.npz`` -- an ordinary ZIP archive (method 0 = STORED,
// never DEFLATEd -- "reading is the operation that has to be fast"), each
// member a standard ``.npy`` array named ``"<key>.npy"``, plus a ``meta``
// member holding a flat JSON object as raw bytes. This header implements
// just enough of both formats to read it back: a STORED-only ZIP central-
// directory parser and a NPY v1.0 header parser -- no dependency on
// Python, numpy, or a general-purpose ZIP/JSON library. Never throws:
// every failure (missing file, corrupt archive, unsupported compression,
// a JSON field this parser doesn't handle, stale metadata) means "no
// sidecar", via read_tpx3scan() returning std::nullopt -- exactly
// pyeventem's own sidecar.read() contract ("missing, stale, truncated...
// all look the same to the caller, which simply rebuilds").
//
// **What this buys**: find_line_checkpoints() (Cheetah.hpp) currently
// re-derives its multi-process split points by scanning the whole file
// once, every single call -- cheap next to a full decode, but not free,
// and pyeventem already has the same information sitting on disk if the
// acquisition has been indexed. Cheetah.hpp tries this sidecar first and
// falls back to its own from-scratch prescan whenever it's missing,
// stale, or doesn't parse -- the fallback is unconditional and
// unconditionally correct either way, so a sidecar this parser can't use
// for any reason costs nothing beyond the (cheap) attempt.

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace tpx3scan
{
    // ------------------------------------------------------------------
    // Minimal STORED-only ZIP reader
    // ------------------------------------------------------------------

    namespace detail
    {
        inline uint16_t read_u16(const uint8_t *p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
        inline uint32_t read_u32(const uint8_t *p) { uint32_t v; std::memcpy(&v, p, 4); return v; }

        // One ZIP central-directory entry this reader cares about: enough
        // to seek to and validate the member's local header, never enough
        // (deliberately) to support anything this narrow use never needs
        // (compressed members, Zip64, multi-disk archives -- all rejected
        // outright below, not silently mishandled).
        struct ZipEntry
        {
            std::string name;
            uint64_t local_header_offset;
            uint64_t uncompressed_size;
            uint16_t compression_method;
        };

        // Whole file read into memory -- a .tpx3scan sidecar is small
        // (pyeventem's own docstring: ~3.4 MB for a 3.2 GB acquisition;
        // scales with scan lines, not file size), so there is no reason
        // to stream it piecewise the way the much larger raw .tpx3 itself
        // has to be.
        inline std::optional<std::vector<uint8_t>> read_whole_file(const std::filesystem::path &path)
        {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) return std::nullopt;
            std::streamsize size = f.tellg();
            if (size <= 0) return std::nullopt;
            f.seekg(0, std::ios::beg);
            std::vector<uint8_t> buf(static_cast<size_t>(size));
            if (!f.read(reinterpret_cast<char *>(buf.data()), size)) return std::nullopt;
            return buf;
        }

        // Locates the End Of Central Directory record (signature
        // PK\x05\x06) by scanning backward from EOF -- np.savez's own
        // writer (Python's zipfile module) never appends an archive
        // comment, so this is normally just the last 22 bytes, but a
        // short backward scan is a cheap safety margin against a stray
        // trailer. Returns the byte offset of the signature, or
        // std::string::npos if not found (not a valid/complete ZIP).
        inline size_t find_eocd(const std::vector<uint8_t> &data)
        {
            static const uint8_t sig[4] = {'P', 'K', 0x05, 0x06};
            const size_t min_eocd = 22;
            if (data.size() < min_eocd) return std::string::npos;
            size_t scan_from = data.size() - min_eocd;
            size_t scan_back = std::min<size_t>(scan_from, 4096); // generous margin
            for (size_t back = 0; back <= scan_back; ++back)
            {
                size_t pos = scan_from - back;
                if (std::memcmp(&data[pos], sig, 4) == 0) return pos;
            }
            return std::string::npos;
        }

        // Parses the central directory into a name -> entry map. Returns
        // an empty vector (not an error type) on anything this reader
        // doesn't support -- Zip64 (>=0xFFFFFFFF fields), multi-disk
        // archives, or a missing/malformed EOCD -- so the caller's own
        // "member not found" handling covers this uniformly.
        inline std::vector<ZipEntry> read_central_directory(const std::vector<uint8_t> &data)
        {
            std::vector<ZipEntry> entries;
            size_t eocd = find_eocd(data);
            if (eocd == std::string::npos) return entries;
            uint16_t n_entries = read_u16(&data[eocd + 10]);
            uint32_t cd_size = read_u32(&data[eocd + 12]);
            uint32_t cd_offset = read_u32(&data[eocd + 16]);
            // Zip64 sentinel values -- refuse rather than misread.
            if (cd_offset == 0xFFFFFFFFu || cd_size == 0xFFFFFFFFu) return {};
            if ((uint64_t)cd_offset + cd_size > data.size()) return {};

            size_t pos = cd_offset;
            static const uint8_t sig[4] = {'P', 'K', 0x01, 0x02};
            for (uint16_t i = 0; i < n_entries; ++i)
            {
                if (pos + 46 > data.size() || std::memcmp(&data[pos], sig, 4) != 0) return {};
                uint16_t compression_method = read_u16(&data[pos + 10]);
                uint32_t uncompressed_size = read_u32(&data[pos + 24]);
                uint16_t name_len = read_u16(&data[pos + 28]);
                uint16_t extra_len = read_u16(&data[pos + 30]);
                uint16_t comment_len = read_u16(&data[pos + 32]);
                uint32_t local_header_offset = read_u32(&data[pos + 42]);
                if (uncompressed_size == 0xFFFFFFFFu || local_header_offset == 0xFFFFFFFFu)
                    return {}; // Zip64 extra field would be needed -- refuse.
                if (pos + 46 + name_len > data.size()) return {};
                std::string name(reinterpret_cast<const char *>(&data[pos + 46]), name_len);
                entries.push_back({std::move(name), local_header_offset, uncompressed_size, compression_method});
                pos += 46 + name_len + extra_len + comment_len;
            }
            return entries;
        }

        // The member's raw (STORED = already-uncompressed) bytes, or
        // std::nullopt if it isn't present, isn't STORED, or its local
        // header doesn't check out.
        inline std::optional<std::vector<uint8_t>> read_member(
            const std::vector<uint8_t> &data, const std::vector<ZipEntry> &entries, const std::string &name)
        {
            for (const auto &e : entries)
            {
                if (e.name != name) continue;
                if (e.compression_method != 0) return std::nullopt; // STORED only
                static const uint8_t sig[4] = {'P', 'K', 0x03, 0x04};
                size_t pos = e.local_header_offset;
                if (pos + 30 > data.size() || std::memcmp(&data[pos], sig, 4) != 0) return std::nullopt;
                uint16_t name_len = read_u16(&data[pos + 26]);
                uint16_t extra_len = read_u16(&data[pos + 28]);
                size_t data_start = pos + 30 + name_len + extra_len;
                if (data_start + e.uncompressed_size > data.size()) return std::nullopt;
                return std::vector<uint8_t>(data.begin() + data_start, data.begin() + data_start + e.uncompressed_size);
            }
            return std::nullopt;
        }

        // ------------------------------------------------------------------
        // Minimal NPY v1.0 reader -- just enough to get a typed element
        // count/pointer out of one of numpy's own arrays. Rejects (returns
        // std::nullopt) anything not a 1-D, C-contiguous, little-endian
        // array with a v1.0 header -- every array this sidecar writes is
        // exactly that (np.save's own default for a plain 1-D array), so
        // this is "reject anything unexpected", not a real limitation.
        // ------------------------------------------------------------------

        struct NpyArray
        {
            std::string descr; // e.g. "<i8", "<u8", "|i1", "|u1"
            size_t itemsize;
            size_t n_elements;
            const uint8_t *data; // points INTO the member buffer passed in -- caller keeps it alive
        };

        inline std::optional<NpyArray> parse_npy(const std::vector<uint8_t> &member)
        {
            static const uint8_t magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
            if (member.size() < 10 || std::memcmp(member.data(), magic, 6) != 0) return std::nullopt;
            uint8_t major = member[6];
            if (major != 1) return std::nullopt; // only v1.0 headers ever occur here (see module docstring)
            uint16_t header_len = read_u16(&member[8]);
            size_t header_start = 10;
            if (header_start + header_len > member.size()) return std::nullopt;
            std::string header(reinterpret_cast<const char *>(&member[header_start]), header_len);

            // Pull 'descr': '<...>' and 'shape': (...) out of the tiny
            // Python-dict-literal header string by substring search --
            // this header is always exactly numpy's own machine-written
            // form (see np.lib.format), never hand-authored, so a full
            // dict-literal parser buys nothing a fixed pair of
            // find()+substr() calls doesn't already cover.
            auto extract = [&](const std::string &key) -> std::optional<std::string> {
                size_t k = header.find("'" + key + "'");
                if (k == std::string::npos) return std::nullopt;
                size_t colon = header.find(':', k);
                size_t quote1 = header.find('\'', colon);
                if (quote1 == std::string::npos) return std::nullopt;
                size_t quote2 = header.find('\'', quote1 + 1);
                if (quote2 == std::string::npos) return std::nullopt;
                return header.substr(quote1 + 1, quote2 - quote1 - 1);
            };
            auto descr = extract("descr");
            if (!descr) return std::nullopt;
            if (header.find("'fortran_order': True") != std::string::npos) return std::nullopt;

            size_t shape_key = header.find("'shape'");
            if (shape_key == std::string::npos) return std::nullopt;
            size_t paren1 = header.find('(', shape_key);
            size_t paren2 = header.find(')', paren1);
            if (paren1 == std::string::npos || paren2 == std::string::npos) return std::nullopt;
            std::string shape_str = header.substr(paren1 + 1, paren2 - paren1 - 1);
            // A 1-D shape is "(N,)"; strip the trailing comma and parse N.
            // (An empty (0-length) array is "(0,)" -- parses to 0 cleanly.)
            size_t comma = shape_str.find(',');
            if (comma == std::string::npos) return std::nullopt;
            std::string n_str = shape_str.substr(0, comma);
            if (n_str.find_first_not_of(" 0123456789") != std::string::npos) return std::nullopt;
            uint64_t n_elements = n_str.empty() ? 0 : std::stoull(n_str);

            size_t itemsize = 0;
            if (descr->size() >= 2 && ((*descr)[1] == 'i' || (*descr)[1] == 'u' || (*descr)[1] == 'f'))
                itemsize = (size_t)std::stoi(descr->substr(2));
            else if (*descr == "|b1" || *descr == "|i1" || *descr == "|u1")
                itemsize = 1;
            if (itemsize == 0) return std::nullopt;

            size_t data_start = header_start + header_len;
            if (data_start + n_elements * itemsize > member.size()) return std::nullopt;
            return NpyArray{*descr, itemsize, (size_t)n_elements, member.data() + data_start};
        }

        // Reinterprets a parsed NpyArray as std::vector<T> -- T's size
        // must exactly match the array's own itemsize (checked; a
        // mismatch, e.g. asking for int64 out of an int8 array by
        // mistake, returns std::nullopt rather than reading garbage).
        template <typename T>
        inline std::optional<std::vector<T>> as_vector(const std::optional<NpyArray> &arr)
        {
            if (!arr || arr->itemsize != sizeof(T)) return std::nullopt;
            std::vector<T> out(arr->n_elements);
            if (arr->n_elements) std::memcpy(out.data(), arr->data, arr->n_elements * sizeof(T));
            return out;
        }

        // One named zip member -> typed vector, in a single call: the
        // member's raw bytes (owned here, in `member_bytes`) MUST outlive
        // the NpyArray view parse_npy() hands back (its `data` pointer
        // points straight into those bytes) -- doing the read+parse+copy
        // as three separate statements (as an earlier version of this
        // file did for exactly two arrays) lets the raw-bytes temporary
        // get destroyed between statements, silently handing as_vector() a
        // dangling pointer that still "successfully" copies whatever
        // memory happens to occupy that address afterward - found the hard
        // way (a differential test against CHEETAH's own from-scratch
        // prescan on real data caught it immediately: every OTHER field
        // matched exactly, only the two fields sourced this way didn't -
        // see Cheetah.hpp's own find_line_checkpoints). Doing the whole
        // read-parse-copy chain inside this one function keeps
        // `member_bytes` alive for exactly as long as it's needed and not
        // one statement longer, so this shape can't recur by accident.
        template <typename T>
        inline std::optional<std::vector<T>> read_member_as_vector(
            const std::vector<uint8_t> &file_bytes, const std::vector<ZipEntry> &entries, const std::string &name)
        {
            auto member_bytes = read_member(file_bytes, entries, name);
            if (!member_bytes) return std::nullopt;
            return as_vector<T>(parse_npy(*member_bytes));
        }

        // The handful of flat scalar fields this reader needs out of the
        // 'meta' member's JSON object (see pyeventem's sidecar.py -- every
        // field is a plain top-level int/float, no nesting). A tiny
        // key-search, not a JSON parser, for the same reason parse_npy()
        // doesn't use a real dict-literal parser: the input is always
        // exactly json.dumps()'s own machine-written form.
        inline std::optional<double> json_number(const std::string &json, const std::string &key)
        {
            std::string needle = "\"" + key + "\":";
            size_t k = json.find(needle);
            if (k == std::string::npos) return std::nullopt;
            size_t start = k + needle.size();
            size_t end = json.find_first_of(",}", start);
            if (end == std::string::npos) return std::nullopt;
            try { return std::stod(json.substr(start, end - start)); }
            catch (...) { return std::nullopt; }
        }
    } // namespace detail

    // ------------------------------------------------------------------
    // The sidecar's contents, in the shape find_line_checkpoints() (see
    // Cheetah.hpp) actually needs to derive checkpoints from -- mirrors
    // pyeventem's RasterSyncTables/ChipLineSegments (raster_sync.py)
    // closely enough to read the same file, not a byte-for-byte port.
    // ------------------------------------------------------------------

    struct ChipLineSegments
    {
        std::vector<int64_t> rise_word_idx;
        std::vector<int64_t> fall_word_idx;
        std::vector<uint64_t> rise_t;
        std::vector<int64_t> line; // continuous across repetitions: 0 .. ny*repetitions-1
    };

    struct Index
    {
        std::array<ChipLineSegments, 4> segments;
        std::vector<int64_t> dt_word_idx;   // word index of the fall event that set this dt
        std::vector<uint64_t> dt_value;     // ToA-tick dwell time effective from that word idx onward
        uint64_t fallback_dt = 0;           // dwell time to use before any dt has been measured
        std::vector<int64_t> header_word_idx; // word index of every chunk-header packet, file order
        std::vector<int8_t> header_chip_id;   // that header's own chip_id, same order/length
    };

    // Reads and validates the sidecar for `tpx3_path` (same rules as
    // pyeventem's own sidecar.read(): format_version, then source file
    // size/mtime, then the scan parameters it was built for -- any
    // mismatch is "no sidecar", not an error). Returns std::nullopt for
    // every failure mode; never throws.
    //
    // `nx`, `dwell_time_ns`: this call's own scan parameters, checked
    // against what the sidecar was built for -- must match exactly (float
    // equality is fine here: both sides pass the same value through with
    // no arithmetic in between, matching pyeventem's own == comparison).
    // `block_words` is deliberately NOT checked (unlike pyeventem's own
    // reader): it only affects how pyeventem's numba kernel re-slices at
    // block boundaries for ITS checkpoint seeding, not the sync tables
    // themselves, which this reader only ever reads back, never re-slices.
    //
    // Source mtime is compared with a tolerance (see the source_mtime_ns
    // check below), not exact equality: converting this platform's
    // std::filesystem write-time to a Unix-epoch nanosecond count (to
    // compare against Python's os.stat().st_mtime_ns, which the sidecar
    // stores) has no portable, exact C++17 conversion, only the standard
    // now()-vs-now() offset trick -- which is only ever accurate to
    // within the gap between two back-to-back clock reads. A few seconds'
    // tolerance costs nothing (source_size still has to match too, and
    // the absolute worst case of a false "fresh" is silently identical to
    // pyeventem's own already-documented, already-accepted same-size-
    // rewrite caveat - see file_cache.py's docstring), while exact
    // equality would make this basically always miss and silently fall
    // back to the full prescan, defeating the entire point.
    inline std::optional<Index> read_tpx3scan(const std::filesystem::path &tpx3_path, int nx, double dwell_time_ns)
    {
        std::filesystem::path sidecar_path = tpx3_path;
        sidecar_path.replace_extension(".tpx3scan");

        std::error_code ec;
        if (!std::filesystem::is_regular_file(tpx3_path, ec) || ec) return std::nullopt;
        auto actual_size = std::filesystem::file_size(tpx3_path, ec);
        if (ec) return std::nullopt;

        auto file_bytes = detail::read_whole_file(sidecar_path);
        if (!file_bytes) return std::nullopt;
        auto entries = detail::read_central_directory(*file_bytes);
        if (entries.empty()) return std::nullopt;

        auto meta_member = detail::read_member(*file_bytes, entries, "meta.npy");
        if (!meta_member) return std::nullopt;
        auto meta_npy = detail::parse_npy(*meta_member);
        if (!meta_npy || meta_npy->itemsize != 1) return std::nullopt;
        std::string meta_json(reinterpret_cast<const char *>(meta_npy->data), meta_npy->n_elements);

        auto format_version = detail::json_number(meta_json, "format_version");
        auto source_size = detail::json_number(meta_json, "source_size");
        auto source_mtime_ns = detail::json_number(meta_json, "source_mtime_ns");
        auto meta_nx = detail::json_number(meta_json, "nx");
        auto meta_dwell = detail::json_number(meta_json, "dwell_time_ns");
        if (!format_version || !source_size || !source_mtime_ns || !meta_nx || !meta_dwell) return std::nullopt;
        // FORMAT_VERSION in pyeventem's sidecar.py, as of this writing --
        // a version this reader doesn't recognize is "no sidecar", the
        // same as pyeventem's own reader treats a future/older version.
        if ((int64_t)*format_version != 1) return std::nullopt;
        if ((uint64_t)*source_size != actual_size) return std::nullopt;
        if ((int)*meta_nx != nx) return std::nullopt;
        if (std::abs(*meta_dwell - dwell_time_ns) > 1e-6) return std::nullopt;

        {
            auto write_time = std::filesystem::last_write_time(tpx3_path, ec);
            if (ec) return std::nullopt;
            auto now_file = std::filesystem::file_time_type::clock::now();
            auto now_sys = std::chrono::system_clock::now();
            auto sys_time = now_sys + std::chrono::duration_cast<std::chrono::system_clock::duration>(write_time - now_file);
            int64_t actual_mtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(sys_time.time_since_epoch()).count();
            int64_t stored_mtime_ns = (int64_t)*source_mtime_ns;
            const int64_t tolerance_ns = 5'000'000'000LL; // 5 s -- see this function's own docstring
            if (std::llabs(actual_mtime_ns - stored_mtime_ns) > tolerance_ns) return std::nullopt;
        }

        Index index;
        auto dt_word_idx = detail::read_member_as_vector<int64_t>(*file_bytes, entries, "dt_word_idx.npy");
        auto dt_value = detail::read_member_as_vector<uint64_t>(*file_bytes, entries, "dt_value.npy");
        if (!dt_word_idx || !dt_value || dt_word_idx->size() != dt_value->size()) return std::nullopt;
        index.dt_word_idx = std::move(*dt_word_idx);
        index.dt_value = std::move(*dt_value);
        auto fallback_dt = detail::json_number(meta_json, "fallback_dt");
        if (!fallback_dt) return std::nullopt;
        index.fallback_dt = (uint64_t)*fallback_dt;

        auto header_word_idx = detail::read_member_as_vector<int64_t>(*file_bytes, entries, "header_word_idx.npy");
        auto header_chip_id = detail::read_member_as_vector<int8_t>(*file_bytes, entries, "header_chip_id.npy");
        if (!header_word_idx || !header_chip_id || header_word_idx->size() != header_chip_id->size())
            return std::nullopt;
        index.header_word_idx = std::move(*header_word_idx);
        index.header_chip_id = std::move(*header_chip_id);

        for (int c = 0; c < 4; ++c)
        {
            std::string prefix = "seg" + std::to_string(c) + "_";
            auto rise_w = detail::read_member_as_vector<int64_t>(*file_bytes, entries, prefix + "rise_word_idx.npy");
            auto fall_w = detail::read_member_as_vector<int64_t>(*file_bytes, entries, prefix + "fall_word_idx.npy");
            auto rise_t = detail::read_member_as_vector<uint64_t>(*file_bytes, entries, prefix + "rise_t.npy");
            auto line = detail::read_member_as_vector<int64_t>(*file_bytes, entries, prefix + "line.npy");
            if (!rise_w || !fall_w || !rise_t || !line) return std::nullopt;
            if (rise_w->size() != fall_w->size() || rise_w->size() != rise_t->size() || rise_w->size() != line->size())
                return std::nullopt;
            index.segments[c].rise_word_idx = std::move(*rise_w);
            index.segments[c].fall_word_idx = std::move(*fall_w);
            index.segments[c].rise_t = std::move(*rise_t);
            index.segments[c].line = std::move(*line);
        }
        return index;
    }

    // One derived checkpoint, matching CHEETAH::find_line_checkpoints's own
    // tuple element order exactly (byte_offset, start_line, dt, rise_t[4],
    // rise_fall[4], line_count[4], chip_id) so Cheetah.hpp can push these
    // straight into the same vector type it already returns.
    struct Checkpoint
    {
        uint64_t byte_offset;
        int start_line;
        uint64_t dt;
        std::array<uint64_t, 4> rise_t;
        std::array<int, 4> rise_fall;   // 0/1, matching the original's own int encoding
        std::array<int, 4> line_count;
        int chip_id;
    };

    namespace detail
    {
        // Rightmost index i such that sorted_keys[i] <= target, or -1. Every
        // array this is called on (word-index columns) is monotonically
        // increasing by construction (both pyeventem's own writer and the
        // ChipLineSegments/dt-table docstrings guarantee it), so a plain
        // binary search applies directly - no need to sort here.
        inline int64_t rightmost_le(const std::vector<int64_t> &sorted_keys, int64_t target)
        {
            int64_t lo = 0, hi = (int64_t)sorted_keys.size() - 1, ans = -1;
            while (lo <= hi)
            {
                int64_t mid = lo + (hi - lo) / 2;
                if (sorted_keys[mid] <= target) { ans = mid; lo = mid + 1; }
                else hi = mid - 1;
            }
            return ans;
        }
    }

    // Derives the same n_splits-1 interior checkpoints CHEETAH::
    // find_line_checkpoints(n_splits) computes by scanning the whole file,
    // straight from an already-parsed sidecar Index - no file access at
    // all. See this header's own module docstring for the state-semantics
    // this mirrors (line_count/rise_fall/rise_t/dt, all as of the exact
    // word position immediately after the slowest chip's own fall for each
    // target line) and Cheetah.hpp's find_line_checkpoints for the
    // ground-truth definition this was differential-tested against.
    //
    // total_lines = ny * repetitions, matching find_line_checkpoints's own
    // `total_lines` exactly - the caller (Cheetah.hpp) already has this.
    inline std::vector<Checkpoint> checkpoints_from_index(const Index &index, int total_lines, int n_splits)
    {
        std::vector<Checkpoint> checkpoints;
        for (int split_idx = 1; split_idx < n_splits; ++split_idx)
        {
            int next_target = (int)(((int64_t)total_lines * split_idx) / n_splits);
            int target_physical_line = next_target - 1; // the line whose completion triggers this checkpoint

            // The checkpoint fires the moment every one of the 4 chips has
            // completed target_physical_line - i.e. at the LATEST (in word
            // order) of their own 4 fall events for it.
            int64_t checkpoint_word = -1;
            bool complete = true;
            for (int c = 0; c < 4 && complete; ++c)
            {
                const auto &seg = index.segments[c];
                // segments[c].line is monotonic and 0-based per-chip - a
                // plain binary search for the exact value (not <=) is
                // enough since every chip credits every line exactly once.
                int64_t lo = 0, hi = (int64_t)seg.line.size() - 1, found = -1;
                while (lo <= hi)
                {
                    int64_t mid = lo + (hi - lo) / 2;
                    if (seg.line[mid] == target_physical_line) { found = mid; break; }
                    else if (seg.line[mid] < target_physical_line) lo = mid + 1;
                    else hi = mid - 1;
                }
                if (found < 0) { complete = false; break; } // this chip never reached that line - can't derive
                checkpoint_word = std::max(checkpoint_word, seg.fall_word_idx[found]);
            }
            if (!complete || checkpoint_word < 0) return {}; // sidecar can't answer this n_splits - caller falls back

            int64_t w = checkpoint_word; // last word fully processed before the checkpoint
            Checkpoint cp{};
            cp.byte_offset = (uint64_t)(w + 1) * 8; // sizeof(event) - see CHEETAH_ADDITIONAL::EVENT
            cp.start_line = next_target;

            int64_t dt_pos = detail::rightmost_le(index.dt_word_idx, w);
            cp.dt = (dt_pos >= 0) ? index.dt_value[dt_pos] : index.fallback_dt;

            for (int c = 0; c < 4; ++c)
            {
                const auto &seg = index.segments[c];
                int64_t j_fall = detail::rightmost_le(seg.fall_word_idx, w);
                cp.line_count[c] = (j_fall >= 0) ? (int)(seg.line[j_fall] + 1) : 0;

                int64_t j_rise = detail::rightmost_le(seg.rise_word_idx, w);
                if (j_rise < 0) { cp.rise_fall[c] = 0; cp.rise_t[c] = 0; }
                else
                {
                    cp.rise_t[c] = seg.rise_t[j_rise];
                    cp.rise_fall[c] = (seg.fall_word_idx[j_rise] > w) ? 1 : 0;
                }
            }

            int64_t h_pos = detail::rightmost_le(index.header_word_idx, w + 1);
            cp.chip_id = (h_pos >= 0) ? (int)index.header_chip_id[h_pos] : 0;

            checkpoints.push_back(cp);
        }
        return checkpoints;
    }
} // namespace tpx3scan

#endif // TPX3SCAN_SIDECAR_HPP
