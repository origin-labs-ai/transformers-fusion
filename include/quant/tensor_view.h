#pragma once

#include "quant/tensor.h"
#include "quant/memory.h"
#include <cstdint>
#include <vector>
#include <cassert>
#include <algorithm>

namespace quant {

class TensorView {
public:
    TensorView(void* data, const Shape& shape, DType dtype = DType::F32)
        : data_(static_cast<char*>(data)), shape_(shape), dtype_(dtype) {
        compute_strides_contiguous();
    }

    TensorView(void* data, const Shape& shape, const std::vector<int64_t>& strides,
               DType dtype = DType::F32)
        : data_(static_cast<char*>(data)), shape_(shape), strides_(strides), dtype_(dtype) {}

    TensorView slice(int dim, int64_t start, int64_t end) const {
        assert(dim >= 0 && dim < shape_.rank);
        assert(start >= 0 && end <= shape_.dims[dim]);

        Shape new_shape = shape_;
        new_shape.dims[dim] = end - start;

        std::vector<int64_t> new_strides = strides_;
        char* new_data = data_ + start * strides_[dim] * dtype_size(dtype_);

        return TensorView(new_data, new_shape, new_strides, dtype_);
    }

    TensorView transpose(int dim1, int dim2) const {
        assert(dim1 >= 0 && dim1 < shape_.rank);
        assert(dim2 >= 0 && dim2 < shape_.rank);

        Shape new_shape = shape_;
        std::swap(new_shape.dims[dim1], new_shape.dims[dim2]);

        std::vector<int64_t> new_strides = strides_;
        std::swap(new_strides[dim1], new_strides[dim2]);

        return TensorView(data_, new_shape, new_strides, dtype_);
    }

    TensorView reshape(const Shape& new_shape) const {
        assert(is_contiguous() && "Can only reshape contiguous views");
        return TensorView(data_, new_shape, dtype_);
    }

    TensorView narrow(int dim, int64_t start, int64_t size) const {
        assert(dim >= 0 && dim < shape_.rank);
        assert(start >= 0 && start + size <= shape_.dims[dim]);

        Shape new_shape = shape_;
        new_shape.dims[dim] = size;

        std::vector<int64_t> new_strides = strides_;
        char* new_data = data_ + start * strides_[dim] * dtype_size(dtype_);

        return TensorView(new_data, new_shape, new_strides, dtype_);
    }

    TensorView unsqueeze(int dim) const {
        assert(dim >= 0 && dim <= shape_.rank);
        assert(shape_.rank + 1 <= 8);

        Shape new_shape;
        new_shape.rank = shape_.rank + 1;
        std::vector<int64_t> new_strides(shape_.rank + 1, 0);

        int si = 0;
        for (int i = 0; i < new_shape.rank; i++) {
            if (i == dim) {
                new_shape.dims[i] = 1;
                new_strides[i] = strides_[si] * shape_.dims[si];
            } else {
                new_shape.dims[i] = shape_.dims[si];
                new_strides[i] = strides_[si];
                si++;
            }
        }

        return TensorView(data_, new_shape, new_strides, dtype_);
    }

    TensorView squeeze(int dim) const {
        assert(dim >= 0 && dim < shape_.rank);
        assert(shape_.dims[dim] == 1);
        assert(shape_.rank - 1 >= 0);

        Shape new_shape;
        new_shape.rank = shape_.rank - 1;
        std::vector<int64_t> new_strides(shape_.rank - 1, 0);

        int si = 0;
        for (int i = 0; i < shape_.rank; i++) {
            if (i == dim) continue;
            new_shape.dims[si] = shape_.dims[i];
            new_strides[si] = strides_[i];
            si++;
        }

        return TensorView(data_, new_shape, new_strides, dtype_);
    }

    template<typename T>
    T at(const std::vector<int64_t>& indices) const {
        // BUGFIX (bug census): no arity/bounds check — OOB read on wrong
        // index count or negative/out-of-range indices. Fail loud instead.
        if ((int)indices.size() != shape_.rank)
            throw Error("TensorView::at: index arity mismatch");
        int64_t offset = 0;
        for (int i = 0; i < shape_.rank; ++i) {
            if (indices[(size_t)i] < 0 || indices[(size_t)i] >= shape_.dims[i])
                throw Error("TensorView::at: index out of bounds");
            offset += indices[(size_t)i] * strides_[i];
        }
        return *reinterpret_cast<T*>(data_ + offset * dtype_size(dtype_));
    }

    template<typename T>
    T* data_at(int64_t flat_offset) {
        if (flat_offset < 0 || flat_offset >= numel())
            throw Error("TensorView::data_at: flat offset out of bounds");
        return reinterpret_cast<T*>(data_ + flat_offset * dtype_size(dtype_));
    }

    template<typename T>
    const T* data_at(int64_t flat_offset) const {
        if (flat_offset < 0 || flat_offset >= numel())
            throw Error("TensorView::data_at: flat offset out of bounds");
        return reinterpret_cast<const T*>(data_ + flat_offset * dtype_size(dtype_));
    }

    bool is_contiguous() const {
        if (shape_.rank == 0) return true;
        int64_t expected_stride = 1;
        for (int i = shape_.rank - 1; i >= 0; --i) {
            if (strides_[i] != expected_stride) return false;
            expected_stride *= shape_.dims[i];
        }
        return true;
    }

    Tensor materialize() const {
        return to_tensor();
    }

    Tensor to_tensor() const {
        Tensor result(shape_, dtype_);
        int64_t total = shape_.numel();
        for (int64_t i = 0; i < total; ++i) {
            int64_t tmp = i;
            int64_t src_offset = 0;
            for (int d = shape_.rank - 1; d >= 0; --d) {
                int64_t idx = tmp % shape_.dims[d];
                tmp /= shape_.dims[d];
                src_offset += idx * strides_[d];
            }
            std::memcpy(result.data<char>() + i * dtype_size(dtype_),
                        data_ + src_offset * dtype_size(dtype_),
                        dtype_size(dtype_));
        }
        return result;
    }

    const Shape& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    int64_t numel() const { return shape_.numel(); }
    int rank() const { return shape_.rank; }
    void* raw_data() { return data_; }
    const void* raw_data() const { return data_; }

private:
    char* data_;
    Shape shape_;
    std::vector<int64_t> strides_;
    DType dtype_;

    void compute_strides_contiguous() {
        strides_.resize(shape_.rank);
        if (shape_.rank == 0) return;
        strides_[shape_.rank - 1] = 1;
        for (int i = shape_.rank - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_.dims[i + 1];
        }
    }
};

} // namespace quant
