#include "quant/tensor.h"
#include "quant/autograd.h"
#include "quant/memory.h"
#include "quant/types.h"
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace quant {

// IEEE 754 FP16 conversion helpers
static inline uint16_t float_to_half(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 31) & 1;
    uint32_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    if (exp == 0) return (uint16_t)(sign << 15);
    if (exp == 0xFF) return (uint16_t)((sign << 15) | 0x7C00 | (mant ? 0x200 : 0));
    int32_t newexp = (int32_t)exp - 127 + 15;
    if (newexp >= 31) return (uint16_t)((sign << 15) | 0x7C00);
    if (newexp <= 0) {
        if (newexp < -10) return (uint16_t)(sign << 15);
        mant = (mant | 0x800000) >> (14 - newexp);
        return (uint16_t)((sign << 15) | (mant >> 1));
    }
    return (uint16_t)((sign << 15) | ((uint32_t)newexp << 10) | (mant >> 13));
}

static inline float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        float v = (float)mant * 0.000000059604644775390625f;
        return sign ? -v : v;
    }
    if (exp == 31) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

Tensor Tensor::to_dtype(DType dtype) const {
    if (dtype_ == dtype) return *this;
    Tensor result(shape_, dtype);
    int64_t n = numel();
    if (dtype_ == DType::F32 && dtype == DType::F16) {
        const float* src = data<float>();
        uint16_t* dst = result.data<uint16_t>();
        for (int64_t i = 0; i < n; i++)
            dst[i] = float_to_half(src[i]);
    } else if (dtype_ == DType::F16 && dtype == DType::F32) {
        const uint16_t* src = data<uint16_t>();
        float* dst = result.data<float>();
        for (int64_t i = 0; i < n; i++)
            dst[i] = half_to_float(src[i]);
    } else {
        QUANT_CHECK(false, "to_dtype: unsupported conversion " +
                  std::to_string((int)dtype_) + " -> " + std::to_string((int)dtype));
    }
    return result;
}

Tensor::Tensor() noexcept
    : shape_(), dtype_(DType::F32), requires_grad_(false),
      grad_(nullptr), offset_(0), is_transposed_(false)
{}

Tensor::Tensor(Shape shape, DType dtype)
    : shape_(shape), dtype_(dtype), requires_grad_(false),
      grad_(nullptr), offset_(0), is_transposed_(false)
{
    size_t bytes = (size_t)shape_.numel() * ::quant::dtype_size(dtype);
    buffer_ = std::make_shared<Buffer>(bytes, 64);
    if (buffer_->data()) memset(buffer_->data(), 0, bytes);
    compute_strides();
}

Tensor::Tensor(Shape shape, std::shared_ptr<Buffer> buffer, DType dtype)
    : shape_(shape), dtype_(dtype), buffer_(buffer),
      requires_grad_(false), grad_(nullptr), offset_(0), is_transposed_(false)
{
    compute_strides();
}

Tensor::~Tensor() noexcept {
    // Self-cleaning autograd registry: only tensors that were ever registered
    // pay the lookup cost. Prevents address-reuse stale entries (flaky exit
    // crashes) from outliving the owning model. registry_alive() is checked
    // first so we never resurrect the singleton during static teardown.
    if (autograd_registered_ && AutogradEngine::registry_alive()) {
        AutogradEngine::instance().unregister_parameter(this);
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), buffer_(other.buffer_),
      requires_grad_(other.requires_grad_),
      grad_(other.grad_ ? std::make_unique<Tensor>(*other.grad_) : nullptr),
      offset_(other.offset_), is_transposed_(other.is_transposed_),
      strides_(other.strides_)
{}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        buffer_ = other.buffer_;
        offset_ = other.offset_;
        strides_ = other.strides_;
        requires_grad_ = other.requires_grad_;
        is_transposed_ = other.is_transposed_;
        grad_ = other.grad_ ? std::make_unique<Tensor>(*other.grad_) : nullptr;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(other.shape_), dtype_(other.dtype_),
      buffer_(std::move(other.buffer_)),
      requires_grad_(other.requires_grad_), grad_(std::move(other.grad_)),
      offset_(other.offset_), is_transposed_(other.is_transposed_),
      strides_(std::move(other.strides_))
{
    other.offset_ = 0;
    other.is_transposed_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        buffer_ = std::move(other.buffer_);
        offset_ = other.offset_;
        strides_ = std::move(other.strides_);
        requires_grad_ = other.requires_grad_;
        is_transposed_ = other.is_transposed_;
        grad_ = std::move(other.grad_);
        other.offset_ = 0;
        other.is_transposed_ = false;
    }
    return *this;
}

void Tensor::compute_strides() {
    strides_.resize(shape_.rank);
    int64_t s = 1;
    for (int i = shape_.rank - 1; i >= 0; i--) {
        strides_[i] = s;
        s *= shape_.dims[i];
    }
    is_transposed_ = false;
}

Tensor Tensor::view(const Shape& new_shape) const {
    QUANT_CHECK(new_shape.numel() == numel(), "view: size mismatch");
    Tensor t;
    t.shape_ = new_shape;
    t.dtype_ = dtype_;
    t.buffer_ = buffer_;
    t.offset_ = offset_;
    t.requires_grad_ = requires_grad_;
    t.is_transposed_ = is_transposed_;
    t.compute_strides();
    return t;
}

Tensor Tensor::slice(int dim, int64_t start, int64_t end) const {
    QUANT_CHECK(dim < shape_.rank, "slice: dim out of range");
    QUANT_CHECK(start >= 0 && end <= shape_.dims[dim], "slice: bounds");
    Tensor t;
    t.shape_ = shape_;
    t.shape_.dims[dim] = end - start;
    t.dtype_ = dtype_;
    t.buffer_ = buffer_;
    t.offset_ = offset_ + (size_t)(start * strides_[dim]) * ::quant::dtype_size(dtype_);
    t.requires_grad_ = requires_grad_;
    t.compute_strides();
    return t;
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    QUANT_CHECK(new_shape.numel() == shape_.numel(), "reshape: size mismatch");
    return view(new_shape);
}

Tensor Tensor::transpose(int dim1, int dim2) const {
    QUANT_CHECK(dim1 < shape_.rank && dim2 < shape_.rank, "transpose: dim out of range");
    if (dim1 == dim2) return *this;
    Tensor t;
    t.shape_ = shape_;
    t.dtype_ = dtype_;
    t.buffer_ = buffer_;
    t.offset_ = offset_;
    t.requires_grad_ = requires_grad_;
    t.strides_ = strides_;
    t.is_transposed_ = false;
    std::swap(t.shape_.dims[dim1], t.shape_.dims[dim2]);
    std::swap(t.strides_[dim1], t.strides_[dim2]);
    if (dim1 == rank()-1 && dim2 == rank()-2 && shape_.dims[dim1] == shape_.dims[dim2]) {
        return t;
    }
    // Clone to contiguous: all math ops assume flat data()[i] indexing
    Tensor copy(t.shape_, t.dtype_);
    copy.requires_grad_ = t.requires_grad_;
    int64_t n = t.numel();
    int64_t elem_bytes = dtype_size(t.dtype_);
    for (int64_t i = 0; i < n; i++) {
        int64_t off = 0, rem = i;
        for (int r = t.rank()-1; r >= 0; --r) {
            int64_t idx = rem % t.shape_.dims[r];
            rem /= t.shape_.dims[r];
            off += idx * t.strides_[r];
        }
        memcpy((char*)copy.data() + i * elem_bytes,
               (const char*)t.data() + off * elem_bytes, elem_bytes);
    }
    return copy;
}

void Tensor::fill(float val) {
    int64_t n = numel();
    switch (dtype_) {
        case DType::F32: {
            float* d = data<float>();
            for (int64_t i = 0; i < n; i++) d[i] = val;
            break;
        }
        case DType::F16: {
            uint16_t h = float_to_half(val);
            uint16_t* d = data<uint16_t>();
            for (int64_t i = 0; i < n; i++) d[i] = h;
            break;
        }
        case DType::I64: {
            int64_t iv = (int64_t)val;
            int64_t* d = data<int64_t>();
            for (int64_t i = 0; i < n; i++) d[i] = iv;
            break;
        }
        case DType::I32: {
            int32_t iv = (int32_t)val;
            int32_t* d = data<int32_t>();
            for (int64_t i = 0; i < n; i++) d[i] = iv;
            break;
        }
        case DType::U8: {
            uint8_t u = (uint8_t)val;
            uint8_t* d = data<uint8_t>();
            for (int64_t i = 0; i < n; i++) d[i] = u;
            break;
        }
        case DType::U4: {
            uint8_t nib = (uint8_t)val & 0xF;
            memset(data(), (int)(nib | (nib << 4)), (size_t)size_bytes());
            break;
        }
    }
}

void Tensor::zero_() {
    if (data()) memset(data(), 0, size_bytes());
}

void Tensor::copy_from(const Tensor& src) {
    QUANT_CHECK(shape_.numel() == src.shape_.numel(), "copy_from: size mismatch");
    memcpy(data(), src.data(), size_bytes());
}

void Tensor::copy_to(Tensor& dst) const {
    dst.copy_from(*this);
}

Tensor Tensor::clone() const {
    Tensor t(shape_, dtype_);
    t.requires_grad_ = requires_grad_;
    if (data() && t.data()) memcpy(t.data(), data(), size_bytes());
    return t;
}

Tensor Tensor::zeros(const Shape& shape, DType dtype) {
    Tensor t(shape, dtype);
    t.zero_();
    return t;
}

Tensor Tensor::ones(const Shape& shape, DType dtype) {
    Tensor t(shape, dtype);
    t.fill(1.0f);
    return t;
}

Tensor Tensor::arange(int64_t n) {
    Tensor t(Shape{n}, DType::F32);
    float* d = t.data<float>();
    for (int64_t i = 0; i < n; i++) d[i] = (float)i;
    return t;
}

int64_t Tensor::offset_to_flat(const std::initializer_list<int64_t>& indices) const {
    int64_t idx = 0;
    int i = 0;
    for (auto ix : indices) {
        QUANT_CHECK(ix < shape_.dims[i], "index out of bounds");
        idx += ix * strides_[i];
        i++;
    }
    return idx;
}

float& Tensor::at(const std::initializer_list<int64_t>& indices) {
    return data<float>()[offset_to_flat(indices)];
}

const float& Tensor::at(const std::initializer_list<int64_t>& indices) const {
    return data<float>()[offset_to_flat(indices)];
}

size_t Tensor::serialized_size() const {
    return sizeof(int32_t) + (size_t)shape_.rank * sizeof(int64_t) + sizeof(uint8_t) + size_bytes();
}

size_t Tensor::serialize(uint8_t* dst) const {
    uint8_t* p = dst;
    int32_t r = (int32_t)shape_.rank;
    memcpy(p, &r, sizeof(r)); p += sizeof(r);
    for (int i = 0; i < shape_.rank; i++) {
        int64_t d = shape_.dims[i];
        memcpy(p, &d, sizeof(d)); p += sizeof(d);
    }
    uint8_t dt = (uint8_t)dtype_;
    memcpy(p, &dt, sizeof(dt)); p += sizeof(dt);
    size_t bytes = size_bytes();
    if (data() && bytes) memcpy(p, data(), bytes);
    p += bytes;
    return (size_t)(p - dst);
}

Tensor Tensor::deserialize(const uint8_t* src, size_t& offset) {
    const uint8_t* p = src + offset;
    int32_t r; memcpy(&r, p, sizeof(r)); p += sizeof(r);
    QUANT_CHECK(r >= 0 && r <= 8, "deserialize: invalid rank");
    Shape shape;
    shape.rank = r;
    for (int i = 0; i < r; i++) {
        int64_t d; memcpy(&d, p, sizeof(d)); p += sizeof(d);
        shape.dims[i] = d;
    }
    uint8_t dt; memcpy(&dt, p, sizeof(dt)); p += sizeof(dt);
    DType dtype = (DType)dt;
    Tensor t(shape, dtype);
    size_t bytes = t.size_bytes();
    if (bytes && t.data()) memcpy(t.data(), p, bytes);
    p += bytes;
    offset = (size_t)(p - src);
    return t;
}

bool Tensor::is_contiguous() const {
    if (is_transposed_) return false;
    if (offset_ != 0) return false;
    int64_t expected = 1;
    for (int i = rank() - 1; i >= 0; i--) {
        if (strides_[i] != expected) return false;
        expected *= shape_.dims[i];
    }
    return true;
}

std::string Tensor::to_string() const {
    std::ostringstream os;
    os << "Tensor" << shape_.to_string() << " " << size_bytes() << "B";
    if (numel() <= 16 && dtype_ == DType::F32) {
        os << " [";
        const float* d = static_cast<const float*>(data());
        for (int64_t i = 0; i < numel(); i++) {
            if (i) os << ", ";
            os << std::fixed << std::setprecision(4) << d[i];
        }
        os << "]";
    }
    return os.str();
}

} // namespace quant
