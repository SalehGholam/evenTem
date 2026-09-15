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

#ifndef CLUSTERRESOLVER_HPP
#define CLUSTERRESOLVER_HPP

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

// Summary statistics for one declustered group of adjacent Timepix3 pixel activations,
// computed by Declusterer::decluster(). kx/ky are the ToT-weighted centroid of the
// cluster's member hits (not the raw seed pixel), tot_sum is the total deposited
// charge (summed Time-over-Threshold) across all member hits, and size is the number
// of raw hits merged into the cluster (including the seed) -- used only by
// resolve_electron_count_lut below; resolve_electron_count ignores it entirely.
struct ClusterInfo
{
    double kx = 0.0;
    double ky = 0.0;
    uint64_t tot_sum = 0;
    int size = 0;
};

// Resolves how many physical electrons a cluster's total deposited charge represents,
// given a calibration constant (the typical summed ToT of a single electron at the
// current beam energy / detector threshold). Rounds to the nearest whole electron and
// never returns a negative count. A cluster far below tot_per_electron (e.g. detector
// noise, X-rays, cosmic rays) resolves to 0 and should not be counted as an electron
// anywhere; a cluster around 2x, 3x, ... tot_per_electron resolves to genuine pile-up.
// This is the original, default resolver -- unchanged, and still what every consumer
// uses unless it's explicitly given an electron_count_lut (see below).
inline int64_t resolve_electron_count(uint64_t tot_sum, double tot_per_electron)
{
    if (tot_per_electron <= 0.0) return 0;
    int64_t n = (int64_t)std::llround((double)tot_sum / tot_per_electron);
    return n > 0 ? n : 0;
}

// Alternative resolver: looks up a cluster's electron count directly from a 2D map
// keyed on (cluster size, total ToT) instead of dividing ToT by a single constant --
// lets a population map fit across both axes (e.g. from decluster.ipynb's GMM +
// hand-edited map) decide the count, rather than assuming counts separate cleanly
// along ToT alone. Row = cluster size, column = ToT, matching
// clustersize_tot_histogram's own [size][tot] layout (see Roi.h etc.) -- a map built
// from that histogram's shape needs no axis reordering. Out-of-range size/tot clamp
// to the table's own edge rather than extrapolating: a table this sparse in its far
// tail has no better answer anyway. This is strictly additive -- resolve_electron_count
// above is untouched and remains the default for every consumer that doesn't opt in.
inline int64_t resolve_electron_count_lut(uint64_t tot_sum, int size,
                                           const std::vector<std::vector<int>>& lut)
{
    if (lut.empty() || lut[0].empty()) return 0;
    int row = std::min(std::max(size, 0), (int)lut.size() - 1);
    int col = (int)std::min(tot_sum, (uint64_t)(lut[0].size() - 1));
    int n = lut[row][col];
    return n > 0 ? n : 0;
}

// Loads a plain-text electron-count template file: lines starting with '#' are
// metadata/comments (beam energy, detector setup, calibration date, source file,
// rank mapping used -- for humans and Python tooling only, never parsed here) and
// are skipped; every other non-empty line is one row of whitespace-separated
// integers, row = cluster size, column = ToT (see resolve_electron_count_lut above).
// Mirrors Cheetah_pixeltrig.hpp's read_patten_file() -- same ifstream-driven,
// dependency-free plain-text approach, no new library.
inline std::vector<std::vector<int>> load_electron_count_lut_file(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Could not open electron_count_lut_file: " + path);

    std::vector<std::vector<int>> lut;
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::vector<int> row;
        int value;
        while (iss >> value) row.push_back(value);
        if (!row.empty()) lut.push_back(std::move(row));
    }
    return lut;
}

#endif // CLUSTERRESOLVER_HPP
