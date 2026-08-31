# API Reference

> **Complete C++ API Documentation for InNova**

---

## 📚 Table of Contents

- [Core Types](#-core-types)
- [Tensor](#-tensor)
- [Autograd](#-autograd)
- [Math Operations](#-math-operations)
- [Transformer](#-transformer)
- [Model](#-model)
- [Backend & Configuration](#-backend--configuration)
- [Tokenizer](#-tokenizer)
- [Sampler](#-sampler)
- [Trainer](#-trainer)
- [Optimizer](#-optimizer)
- [QUANT Format](#-quant-format)
- [Quantization](#-quantization)
- [MoE Variants](#-moe-variants)
- [GPU Compute](#-gpu-compute)
- [Memory Management](#-memory-management)

> **Per-file API details:** For implementation-level API documentation, see the **[wiki/files/](../wiki/files/_index.md)** directory with per-file docs for every source file.

---

## 🏷️ Core Types

### Namespace

All InNova APIs are in the `quant` namespace:

```cpp
namespace quant {
    // All API types and functions
}

using namespace quant; // Recommended for simplicity
```

### Enumerations

#### Format

Supported quantization formats:

```cpp
enum class Format : uint8_t {
    QUANT1            = 0,  // 0.5 BPW stored FP16 block means; 1.0 is the documented conservative cap
    QUANT_Q0        = 1,  // 1.50 BPW, sign + shared FP16 scale (lossy)
    QUANT2            = 2,  // 2.00 BPW, 4 centroids Lloyd-Max (lossy)
    QUANT4            = 3,  // 4.00 BPW, 16 centroids Lloyd-Max (lossy)
    QUANT8            = 4,  // 8.00 BPW, 256 centroids Lloyd-Max (lossy)
    QUANT16           = 5,  // 16.00 BPW, FP16 storage (lossy)
    QUANT32           = 6,  // 32.00 BPW, FP32 identity (lossless)
    QUANT1_G        = 7,  // 1.0 BPW, block FP16 scale + sign bits (no per-64 zp/scale)
    QUANT_Q0_G    = 8,  // 1.5 BPW, block FP16 scale + sign + refinement
    QUANT2_G        = 9,  // 2.5 BPW, 2-bit lattice + per-64-group FP16 zp/scale
    QUANT4_G        = 10, // 4.5 BPW, 4-bit lattice + per-64-group FP16 zp/scale
    QUANT8_G        = 11, // 8.5 BPW, 8-bit lattice + per-64-group FP16 range/zp
    QUANT16_G       = 12, // 16.0 BPW, FP16 storage (same as QUANT16; no grouping at 16 BPW)
    QUANT_Q1     = 13, // 2.0 BPW, lossy sparse (uint16 index, int8 value)
    QUANT_Q1_G = 14, // 2.0 BPW, lossy grouped sparse (per-half-block FP16 scale)
};

// Usage
Format fmt = Format::QUANT8;
const char* name = format_name(fmt);  // "QUANT8"
float bpw = format_bpw(fmt);          // 8.0
```

#### DType

Supported data types:

```cpp
enum class DType : uint8_t {
    I64,    // int64_t
    U8,     // uint8_t
    U4,     // 4-bit packed (2 per byte)
    I2,     // 2-bit ternary packed (4 per byte)
    I1,     // 1-bit binary packed (8 per byte)
    F16,    // half precision
    F32     // single precision
};

// Usage
DType dt = DType::F32;
size_t size = dtype_size(dt);  // 4 bytes
```

#### Device

Compute device types:

```cpp
enum class Device : uint8_t {
    CPU,
    GPU
};
```

#### Precision

Compute precision:

```cpp
enum class Precision : uint8_t {
    FP32,
    FP16,
    INT8,
    MIXED
};
```

---

## 🎯 Tensor

The fundamental data structure in InNova.

### Class: Tensor

```cpp
class Tensor {
public:
    // Construction
    Tensor();
    Tensor(const Shape& shape, DType dtype = DType::F32);
    Tensor(const Shape& shape, DType dtype, Allocator* alloc);
    
    // Static constructors
    static Tensor zeros(const Shape& shape, DType dtype = DType::F32);
    static Tensor ones(const Shape& shape, DType dtype = DType::F32);
    static Tensor empty(const Shape& shape, DType dtype = DType::F32);
    static Tensor from_data(const Shape& shape, DType dtype, const void* data);
    
    // Accessors
    const Shape& shape() const;
    int64_t dim(int index) const;
    int rank() const;
    int64_t numel() const;
    DType dtype() const;
    size_t nbytes() const;
    
    // Data access
    template <typename T>
    T* data();
    template <typename T>
    const T* data() const;
    template <typename T>
    T& operator()(const std::vector<int64_t>& indices);
    template <typename T>
    const T& operator()(const std::vector<int64_t>& indices) const;
    
    // Properties
    bool is_contiguous() const;
    bool requires_grad() const;
    void requires_grad(bool requires);
    
    // Shape manipulation
    Tensor reshape(const Shape& new_shape) const;
    Tensor permute(const std::vector<int>& dims) const;
    Tensor transpose(int dim0 = 0, int dim1 = 1) const;
    Tensor contiguous() const;
    
    // Slicing
    Tensor slice(int dim, int64_t start, int64_t end) const;
    Tensor index_select(int dim, const Tensor& indices) const;
    
    // Views
    Tensor view(const Shape& shape) const;
    Tensor narrow(int dim, int64_t start, int64_t length) const;
    
    // Serialization
    void save(std::ostream& os) const;
    static Tensor load(std::istream& is);
    
    // Utility
    void fill_(float value);
    void zero_();
    
    // Autograd support
    void backward(const Tensor& gradient = Tensor());
};
```

### Shape

```cpp
class Shape {
public:
    std::vector<int64_t> dims;
    
    Shape();
    Shape(const std::vector<int64_t>& dims);
    Shape(std::initializer_list<int64_t> dims);
    
    int rank() const;
    int64_t numel() const;
    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const;
};

// Usage
Shape s1{2, 3, 4};           // 2x3x4 tensor
Shape s2 = {2, 3, 4};         // Same
Tensor t(s1, DType::F32);    // Create tensor with shape
```

---

## ⚙️ Autograd

Automatic differentiation engine.

### Class: AutogradEngine

```cpp
class AutogradEngine {
public:
    // Engine control
    static void set_enabled(bool enabled);
    static bool enabled();
    static AutogradEngine& instance();
    
    // Parameter management
    void register_parameter(Tensor* param);
    void unregister_parameter(Tensor* param);
    bool is_parameter(Tensor* tensor) const;
    
    // Computation
    void clear();
    void backward(const Tensor& loss);
    
    // Operation recording
    template <typename... Args>
    Tensor record(const std::string& op_name, Tensor&& result, Args&&... args);
    
    // Operations (with autograd support)
    static Tensor matmul_op(const Tensor& a, const Tensor& b);
    static Tensor matmul_op(const Tensor& a, const Tensor& b, 
                            int64_t M, int64_t N, int64_t K);
    static Tensor embedding_op(const Tensor& input_ids, const Tensor& weight);
    static Tensor bias_add_op(const Tensor& input, const Tensor& bias);
    static Tensor layer_norm_op(const Tensor& input, const Tensor& weight, 
                                 const Tensor& bias, float eps);
    static Tensor cross_entropy_op(const Tensor& logits, const Tensor& targets);
    static Tensor add_op(const Tensor& a, const Tensor& b);
    static Tensor mul_op(const Tensor& a, const Tensor& b);
    static Tensor relu_op(const Tensor& input);
    static Tensor gelu_op(const Tensor& input);
    static Tensor softmax_op(const Tensor& input, int dim = -1);
    static Tensor sum_op(const Tensor& input, int dim = -1);
    static Tensor mean_op(const Tensor& input, int dim = -1);
    
    // Parameter access
    const std::unordered_set<Tensor*>& parameters() const;
};
```

### Usage

```cpp
// Enable autograd
AutogradEngine::set_enabled(true);

// Create parameters
Tensor w = Tensor::zeros({10, 10}, DType::F32);
Tensor b = Tensor::zeros({10}, DType::F32);
w.requires_grad(true);
b.requires_grad(true);

// Register parameters
auto& engine = AutogradEngine::instance();
engine.register_parameter(&w);
engine.register_parameter(&b);

// Forward pass with autograd operations
Tensor x = /* input */;
Tensor y = AutogradEngine::matmul_op(x, w);
Tensor z = AutogradEngine::bias_add_op(y, b);

// Compute loss
Tensor loss = /* compute loss */;

// Backward pass
engine.backward(loss);

// Access gradients
Tensor* dw = /* get gradient for w */;
Tensor* db = /* get gradient for b */;

// Clear for next iteration
engine.clear();
```

---

## 🔢 Math Operations

Mathematical operations with AVX2 optimization.

### Namespace: quant::math

```cpp
namespace quant::math {
    // BLAS operations
    void gemm(float alpha, const Tensor& A, const Tensor& B, 
              float beta, Tensor& C);
    void gemv(float alpha, const Tensor& A, const Tensor& x, 
              float beta, Tensor& y);
    
    // Element-wise operations
    void add(const Tensor& a, const Tensor& b, Tensor& out);
    void sub(const Tensor& a, const Tensor& b, Tensor& out);
    void mul(const Tensor& a, const Tensor& b, Tensor& out);
    void div(const Tensor& a, const Tensor& b, Tensor& out);
    
    // Reductions
    void sum(const Tensor& input, Tensor& out, int dim = -1);
    void mean(const Tensor& input, Tensor& out, int dim = -1);
    void max(const Tensor& input, Tensor& out, int dim = -1);
    void min(const Tensor& input, Tensor& out, int dim = -1);
    
    // Activation functions
    void relu(const Tensor& input, Tensor& out);
    void gelu(const Tensor& input, Tensor& out);
    void softmax(const Tensor& input, Tensor& out, int dim = -1);
    void layer_norm(const Tensor& input, const Tensor& weight, 
                    const Tensor& bias, Tensor& out, float eps);
    
    // Other
    void transpose(const Tensor& input, Tensor& out);
    void matmul(const Tensor& a, const Tensor& b, Tensor& out);
    
    // In-place operations
    void add_(Tensor& a, const Tensor& b);
    void sub_(Tensor& a, const Tensor& b);
    void mul_(Tensor& a, const Tensor& b);
    void div_(Tensor& a, const Tensor& b);
}
```

---

## 🤖 Transformer

Transformer architecture implementation.

### Class: Embedding

```cpp
class Embedding {
public:
    Tensor weight;
    
    Embedding(int64_t vocab_size, int64_t dim);
    Tensor forward(const Tensor& input_ids) const;
    size_t param_count() const;
};
```

### Class: Linear

```cpp
class Linear {
public:
    Tensor weight;
    Tensor bias;
    
    Linear(int64_t in_features, int64_t out_features);
    Tensor forward(const Tensor& input) const;
    size_t param_count() const;
};
```

### Class: Attention

```cpp
class Attention {
public:
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear o_proj;
    float scale;
    int n_heads;
    int head_dim;
    
    Attention(int64_t dim, int64_t n_heads);
    Tensor forward(const Tensor& x, const Tensor& positions = Tensor()) const;
    size_t param_count() const;
};
```

### Class: FeedForward

```cpp
class FeedForward {
public:
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
    
    FeedForward(int64_t dim, int64_t hidden_dim);
    Tensor forward(const Tensor& x) const;
    size_t param_count() const;
};
```

### Class: TransformerBlock

```cpp
class TransformerBlock {
public:
    Attention attention;
    Linear attention_norm;
    FeedForward ffn;
    Linear ffn_norm;
    
    TransformerBlock(int64_t dim, int64_t n_heads, int64_t hidden_dim);
    Tensor forward(const Tensor& x, const Tensor& positions = Tensor()) const;
    size_t param_count() const;
};
```

### Class: TransformerConfig

```cpp
struct TransformerConfig {
    int64_t dim = 512;
    int64_t n_layers = 6;
    int64_t n_heads = 8;
    int64_t hidden_dim = 2048;
    int64_t vocab_size = 50257;
    float norm_eps = 1e-6f;
    float dropout = 0.1f;
};
```

### Class: DenseModel

```cpp
class DenseModel : public Model {
public:
    Embedding* tok_embeddings;
    std::vector<TransformerBlock*> layers;
    Linear* norm;
    Linear* lm_head;
    
    DenseModel(const TransformerConfig& config);
    ~DenseModel();
    
    Tensor forward(const Tensor& input_ids, const Tensor& positions) const override;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;
    size_t param_count() const override;
};
```

---

## 📦 Model

Base class for all models and model loading/saving.

### Class: Model

```cpp
class Model {
public:
    virtual ~Model();
    
    // Forward pass
    virtual Tensor forward(const Tensor& input_ids, const Tensor& positions = Tensor()) const = 0;
    
    // Save/Load
    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;
    
    // Parameter count
    virtual size_t param_count() const = 0;
    
    // Factory methods
    static Model* load(const std::string& path, const BackendConfig& config = BackendConfig());
    static DenseModel* create_dense(const TransformerConfig& config);
};
```

### Class: BackendConfig

```cpp
struct BackendConfig {
    Device device = Device::CPU;
    Precision precision = Precision::FP32;
    int n_threads = 0;  // 0 = auto
};
```

### Class: Backend

```cpp
class Backend {
public:
    Backend(const BackendConfig& config = BackendConfig());
    
    // Device information
    Device device() const;
    Precision precision() const;
    int n_threads() const;
    
    // Allocation
    Tensor allocate(const Shape& shape, DType dtype);
    
    // Synchronization
    void synchronize() const;
};
```

---

## 🔤 Tokenizer

Byte-Pair Encoding (BPE) tokenizer.

### Class: Tokenizer

```cpp
class Tokenizer {
public:
    Tokenizer();
    Tokenizer(const std::string& vocab_path);
    Tokenizer(const std::vector<std::string>& vocab);
    
    // Tokenization
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    
    // Vocabulary
    size_t vocab_size() const;
    const std::string& get_token(int id) const;
    int get_token_id(const std::string& token) const;
    
    // Special tokens
    int pad_id() const;
    int bos_id() const;
    int eos_id() const;
    int unk_id() const;
    
    // Batch operations
    Tensor encode_batch(const std::vector<std::string>& texts) const;
    std::vector<std::string> decode_batch(const Tensor& tokens) const;
    
    // Save/Load
    void save(const std::string& path) const;
    static Tokenizer load(const std::string& path);
};
```

### Usage

```cpp
// Create tokenizer
Tokenizer tokenizer("tokenizer.json");

// Encode text
std::vector<int> tokens = tokenizer.encode("Hello, world!");

// Decode tokens
std::string text = tokenizer.decode(tokens);

// Batch encode
std::vector<std::string> texts = {"Hello", "World"};
Tensor batch_tokens = tokenizer.encode_batch(texts);
```

---

## 🎲 Sampler

Token sampling strategies.

### Class: Sampler

```cpp
class Sampler {
public:
    Sampler(float temperature = 1.0f, int top_k = 50, float top_p = 1.0f);
    
    // Sampling
    int sample(const Tensor& logits) const;
    std::vector<int> sample_batch(const Tensor& logits, int n_samples) const;
    
    // Configuration
    float temperature() const;
    void set_temperature(float temperature);
    int top_k() const;
    void set_top_k(int top_k);
    float top_p() const;
    void set_top_p(float top_p);
    
    // Reproducibility
    void set_seed(int64_t seed);
};
```

### Usage

```cpp
// Create sampler
Sampler sampler(0.7f, 50, 0.95f);

// Sample from logits
Tensor logits = /* model output */;
int next_token = sampler.sample(logits);

// Batch sampling
std::vector<int> samples = sampler.sample_batch(logits, 5);
```

---

## 🎓 Trainer

Training infrastructure.

### Class: DataLoader

```cpp
class DataLoader {
public:
    DataLoader(Tokenizer* tokenizer, const std::string& data_path,
               int64_t batch_size, int64_t seq_length);
    
    bool next_batch(Tensor& input_ids, Tensor& labels);
    void shuffle();
    void reset();
    int64_t num_batches() const;
};
```

### Class: TrainConfig

```cpp
struct TrainConfig {
    int64_t batch_size = 4;
    int64_t seq_length = 128;
    int64_t num_epochs = 10;
    float learning_rate = 0.001f;
    int64_t log_interval = 10;
    int64_t save_interval = 100;
    std::string output_path = "trained.quant";
};
```

### Class: TrainMetrics

```cpp
struct TrainMetrics {
    float loss = 0.0f;
    float perplexity = 0.0f;
    float learning_rate = 0.0f;
    int64_t step = 0;
};
```

### Class: Trainer

```cpp
class Trainer {
public:
    using LogCallback = std::function<void(const TrainMetrics&)>;
    
    Trainer(Model* model, Tokenizer* tokenizer);
    
    // Setup
    void compile(Optimizer* optimizer);
    
    // Training
    void fit(DataLoader& dataloader, const TrainConfig& config);
    float train_step(const Tensor& input_ids, const Tensor& labels);
    
    // Checkpointing
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    
    // Logging
    void set_log_callback(LogCallback callback);
    const TrainMetrics& metrics() const;
    
private:
    Model* model_;
    Tokenizer* tokenizer_;
    Optimizer* optimizer_;
    std::vector<Tensor*> model_params_;
    int64_t step_;
    TrainMetrics metrics_;
    LogCallback log_cb_;
};
```

---

## 🔄 Optimizer

Optimization algorithms.

### Class: Optimizer

```cpp
class Optimizer {
public:
    virtual ~Optimizer();
    
    virtual void add_param_group(const std::vector<Tensor*>& params) = 0;
    virtual void zero_grad() = 0;
    virtual void step() = 0;
    virtual float get_lr() const = 0;
    virtual void set_lr(float lr) = 0;
};
```

### Class: SGD

```cpp
class SGD : public Optimizer {
public:
    SGD(float lr, float momentum = 0.9f, float weight_decay = 0.0f);
    
    void add_param_group(const std::vector<Tensor*>& params) override;
    void zero_grad() override;
    void step() override;
    float get_lr() const override;
    void set_lr(float lr) override;
    
private:
    float lr_;
    float momentum_;
    float weight_decay_;
    std::vector<Tensor> momentums_;
};
```

### Class: Adam

```cpp
class Adam : public Optimizer {
public:
    Adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, 
         float eps = 1e-8f, float weight_decay = 0.0f);
    
    void add_param_group(const std::vector<Tensor*>& params) override;
    void zero_grad() override;
    void step() override;
    float get_lr() const override;
    void set_lr(float lr) override;
    
private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    int step_;
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
};
```

### Class: AdamW

```cpp
class AdamW : public Optimizer {
public:
    AdamW(float lr, float beta1 = 0.9f, float beta2 = 0.999f, 
          float eps = 1e-8f, float weight_decay = 0.01f);
    
    void add_param_group(const std::vector<Tensor*>& params) override;
    void zero_grad() override;
    void step() override;
    float get_lr() const override;
    void set_lr(float lr) override;
};
```

---

## 💾 QUANT Format

Binary format for model storage.

### Class: QUANTReader

```cpp
class QUANTReader {
public:
    QUANTReader(std::istream& is);
    QUANTReader(const std::string& path);
    
    bool valid() const;
    Format format() const;
    Shape shape() const;
    DType dtype() const;
    
    void read_header();
    void read_metadata();
    Tensor read_tensor();
    
    template <typename T>
    std::vector<T> read_array(size_t count);
    
    void seek(size_t pos);
    size_t tell() const;
};
```

### Class: QUANTWriter

```cpp
class QUANTWriter {
public:
    QUANTWriter(std::ostream& os);
    QUANTWriter(const std::string& path);
    
    void write_header(Format format);
    void write_metadata(const std::unordered_map<std::string, std::string>& metadata);
    void write_tensor(const Tensor& tensor);
    
    template <typename T>
    void write_array(const std::vector<T>& data);
    
    size_t tell() const;
};
```

### File Format

```
+------------------+
| Header (16 B)  |  Magic: "QUANT\0", Version, Flags
+------------------+
| Metadata (Var)  |  Key-value pairs (JSON-like)
+------------------+
| Weight Blocks   |  Mixed format weight data
+------------------+
| Codebooks       |  For QUANT8/QUANT4 formats
+------------------+
```

---

## 📉 Quantization

### Class: Quantizer

```cpp
class Quantizer {
public:
    virtual ~Quantizer();
    virtual Tensor quantize(const Tensor& input) = 0;
    virtual Tensor dequantize(const Tensor& input) = 0;
};
```

### Class: STEQuantizer

```cpp
class STEQuantizer : public Quantizer {
public:
    STEQuantizer(Format format, int codebook_size = 256);
    
    Tensor quantize(const Tensor& input) override;
    Tensor dequantize(const Tensor& input) override;
    
    const Tensor& codebook() const;
    void update_codebook(const Tensor& gradients);
};
```

### Class: FormatPlanner

```cpp
class FormatPlanner {
public:
    FormatPlanner(float target_bpw = 1.50f);
    
    void analyze(const Tensor& weights, const Tensor& activations);
    Format get_format(int64_t weight_idx) const;
    float get_score(int64_t weight_idx) const;
    
    void set_target_bpw(float bpw);
    float get_target_bpw() const;
};
```

---

## 🎯 MoE Variants

### Class: MoEConfig

```cpp
struct MoEConfig {
    int64_t num_experts = 8;
    int64_t top_k = 2;
    std::string routing = "softmax";
};
```

### Class: Expert

```cpp
class Expert {
public:
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
    
    Expert(int64_t dim, int64_t hidden_dim);
    Tensor forward(const Tensor& x) const;
};
```

### MoE Functions

```cpp
// Routing
Tensor softmax_with_topk(const Tensor& logits, int64_t k, 
                          Tensor& indices_out, Tensor& weights_out);
int64_t hash_token(int64_t token_id, int64_t range);
float compute_load_balance_loss(const Tensor& router_logits, 
                                const Tensor& expert_indices, 
                                int64_t num_experts);

// Dispatch
Tensor moe_forward(const std::vector<Expert*>& experts,
                   const Tensor& x,
                   const Tensor& expert_indices,
                   const Tensor& expert_weights);
```

---

## 🖥️ GPU Compute

### Class: GPUContext

```cpp
class GPUContext {
public:
    GPUContext();
    ~GPUContext();
    
    bool is_available() const;
    const std::string& device_name() const;
    size_t memory() const;
    
    // Buffer management
    GPUBuffer create_buffer(size_t size, const void* data = nullptr);
    void release_buffer(GPUBuffer& buffer);
    
    // Upload/Download
    void upload_buffer(GPUBuffer& buffer, const void* data, size_t size);
    void download_buffer(const GPUBuffer& buffer, void* data, size_t size);
};
```

### Class: GPUBuffer

```cpp
class GPUBuffer {
public:
    GPUBuffer();
    size_t size() const;
    void* map();
    void unmap();
};
```

### GPU Operations

```cpp
namespace quant::gpu {
    void gemm(GPUContext& ctx, float alpha, const GPUBuffer& A, const GPUBuffer& B,
              float beta, GPUBuffer& C, int64_t M, int64_t N, int64_t K);
    void gemv(GPUContext& ctx, float alpha, const GPUBuffer& A, const GPUBuffer& x,
              float beta, GPUBuffer& y, int64_t M, int64_t N);
    void relu(GPUContext& ctx, const GPUBuffer& x, GPUBuffer& y, int64_t N);
    void gelu(GPUContext& ctx, const GPUBuffer& x, GPUBuffer& y, int64_t N);
}
```

---

## 🧠 Memory Management

### Class: Allocator

```cpp
class Allocator {
public:
    virtual ~Allocator();
    virtual void* allocate(size_t size, size_t alignment = 64) = 0;
    virtual void deallocate(void* ptr, size_t size, size_t alignment = 64) = 0;
};
```

### Class: ArenaAllocator

```cpp
class ArenaAllocator : public Allocator {
public:
    ArenaAllocator(size_t capacity = 1024 * 1024 * 1024);
    ~ArenaAllocator();
    
    void* allocate(size_t size, size_t alignment = 64) override;
    void deallocate(void* ptr, size_t size, size_t alignment = 64) override;
    
    void reset();
    size_t used() const;
    size_t capacity() const;
};
```

---

## 🔥 Utility Functions

### Error Handling

```cpp
namespace quant {
    void QUANT_CHECK(bool condition, const std::string& message);
    void QUANT_ASSERT(bool condition, const std::string& message);
};
```

### Random Number Generation

```cpp
class RNG {
public:
    RNG(uint64_t seed = 42);
    
    float uniform();
    float normal();
    int64_t uniform_int(int64_t low, int64_t high);
    
    void set_seed(uint64_t seed);
    uint64_t get_seed() const;
};
```

---

## 📚 Index

### Classes

- [AutogradEngine](#class-autogradengine)
- [Backend](#class-backend)
- [BackendConfig](#class-backendconfig)
- [DataLoader](#class-dataloader)
- [DenseModel](#class-densemodel)
- [Embedding](#class-embedding)
- [Expert](#class-expert)
- [FormatPlanner](#class-formatplanner)
- [GPUBuffer](#class-gpubffer)
- [GPUContext](#class-gpucontext)
- [Linear](#class-linear)
- [Model](#class-model)
- [QUANTReader](#class-quantreader)
- [QUANTWriter](#class-quantwriter)
- [Optimizer](#class-optimizer)
- [Quantizer](#class-quantizer)
- [RNG](#class-rng)
- [Sampler](#class-sampler)
- [SGD](#class-sgd)
- [Shape](#class-shape)
- [STEQuantizer](#class-stequantizer)
- [Tensor](#class-tensor)
- [Tokenizer](#class-tokenizer)
- [Trainer](#class-trainer)
- [TransformerBlock](#class-transformerblock)

### Namespaces

- [quant](#-namespace)
- [quant::math](#namespace-quantmath)
- [quant::gpu](#namespace-quantgpu)

### Enumerations

- [Device](#device)
- [DType](#dtype)
- [Format](#format)
- [Precision](#precision)

---

*Last updated: July 12, 2026*
