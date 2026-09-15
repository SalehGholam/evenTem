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

#ifndef ROI4D_HPP
#define ROI4D_HPP

#include <vector>
#include <cstdint>

class Roi4DBase {
public:
    virtual ~Roi4DBase() = default;
    virtual void init(int L_1, int L_0, int n_cam, int det_bin) = 0;
    virtual void* getData() = 0;
    virtual void increment(int,int,int,int) = 0;
    // Adds n (rather than 1) to one voxel -- used by Roi's charge-weighted declustered
    // extraction, where a single resolved cluster can represent more than one electron.
    // Like increment(), this truncates/wraps at the underlying T's bit depth if the
    // voxel's count exceeds it -- same known limitation as the raw per-event path.
    virtual void increment_by(int,int,int,int,uint64_t) = 0;
    virtual size_t dim0() const = 0;
    virtual size_t dim1() const = 0;
    virtual size_t dim2() const = 0;
    virtual size_t dim3() const = 0;

};

// Backing store is one flat, contiguous vector<T> (row-major over
// [L_1][L_0][n_cam/det_bin][n_cam/det_bin]) instead of a 4-level nested
// vector<vector<vector<vector<T>>>>. The nested form allocated one small heap chunk
// per innermost row -- L_1*L_0*(n_cam/det_bin) separate allocations for a full-frame
// ROI (millions for a 512x512 scan) -- which was slow to allocate/zero, scattered
// pointer-chasing on every per-event increment, and required walking the whole
// structure element-by-element into a temporary buffer before it could be copied into
// numpy (see Roi::create_py_array, now a single memcpy). A single flat allocation
// fixes all three; the (x,y,kx,ky) -> flat index math below is the only behavior
// change, values and shapes are identical.
template <typename T>
class Roi4D : public Roi4DBase {
public:
    std::vector<T> data;

    void init(int L_1, int L_0, int n_cam, int det_bin) {
        d0 = L_1;
        d1 = L_0;
        d2 = n_cam / det_bin;
        d3 = n_cam / det_bin;
        data.assign(d0 * d1 * d2 * d3, 0);
    }
    void* getData() override {
        return &data;
    }
    inline size_t index(int x, int y, int kx, int ky) const {
        return ((static_cast<size_t>(x) * d1 + y) * d2 + kx) * d3 + ky;
    }
    void increment(int x, int y, int kx, int ky) override {
        data[index(x, y, kx, ky)]++;
    }
    void increment_by(int x, int y, int kx, int ky, uint64_t n) override {
        data[index(x, y, kx, ky)] += (T)n;
    }
    size_t dim0() const override { return d0; }
    size_t dim1() const override { return d1; }
    size_t dim2() const override { return d2; }
    size_t dim3() const override { return d3; }

private:
    size_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
};

#endif // ROI4D_HPP
