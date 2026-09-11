# Usage Guide

> **How to Use Transcender for Inference and Training**
>
> **Production-hardening sync (2026-09-11):** version **1.1.0 / R0001.01**, **105 Q-series formats**
> (`FORMAT_COUNT=105`, `include/quant/types.h:67`, no TWI), index magic **TranscenderIDX**
> (`src/codec/quant_format.cpp:546`). Tool binaries use underscores
> (`quant_infer`, `quant_train`, … — `CMakeLists.txt:344-393`), not dashes. Example
> numbers below (dimensions, parameter counts, MB sizes, BPW splits) are **UNVERIFIED**
> illustrative outputs unless they trace to `bench_format_comparison.csv`.

---

## 🎯 Quick Start Examples

### Run Inference

```bash
# Basic inference with a model
./build/bin/quant-infer --model path/to/model.quant --prompt "Hello, world!"

# With more options
./build/bin/quant-infer \
    --model path/to/model.quant \
    --prompt "Write a poem about AI" \
    --max-tokens 100 \
    --temperature 0.7 \
    --top-k 50 \
    --output result.txt
```

### Train a Model

```bash
# Train from scratch
./build/bin/quant-train \
    --config path/to/config.json \
    --data path/to/training.txt \
    --output path/to/trained.quant \
    --batch-size 4 \
    --seq-length 128 \
    --epochs 10 \
    --learning-rate 0.001
```

### Fine-tune a Model

```bash
# Fine-tune an existing model
./build/bin/quant-finetune \
    --model path/to/base.quant \
    --data path/to/fine-tune.txt \
    --output path/to/fine-tuned.quant \
    --epochs 3 \
    --learning-rate 1e-5
```

### Convert Model Formats

```bash
# Convert HuggingFace safetensors to QUANT
./build/bin/quant-convert \
    --input model.safetensors \
    --output model.quant \
    --target-bpw 1.50

# Get model info
./build/bin/quant-info --model model.quant
```

### Run Benchmarks

```bash
# Benchmark inference speed
./build/bin/quant-bench \
    --model model.quant \
    --prompts bench/prompts.txt \
    --iterations 100 \
    --warmup 10
```

---

## 📚 Command-Line Tools Reference

### quant-infer - Run Inference

**Description:** Run inference with a trained Transcender model.

**Usage:**
```bash
quant-infer [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model`, `-m` | string | required | Path to the model file (.quant) |
| `--prompt`, `-p` | string | required | Input prompt for inference |
| `--max-tokens`, `-t` | int | 256 | Maximum number of tokens to generate |
| `--temperature` | float | 1.0 | Temperature for sampling (0.0-2.0) |
| `--top-k` | int | 50 | Top-k sampling |
| `--top-p` | float | 1.0 | Top-p (nucleus) sampling |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--batch-size` | int | 1 | Batch size for inference |
| `--output`, `-o` | string | stdout | Output file for generated text |
| `--verbose`, `-v` | flag | false | Verbose output |
| `--gpu` | flag | false | Use GPU acceleration (if available) |
| `--precision` | string | auto | Precision: auto, fp32, fp16, int8 |

**Examples:**

```bash
# Basic inference
quant-infer -m model.quant -p "Once upon a time"

# Creative writing with temperature
quant-infer -m model.quant -p "Write a haiku" --temperature 0.8 --max-tokens 50

# Batch inference
quant-infer -m model.quant -p "Prompt 1" -p "Prompt 2" --batch-size 2

# Save output to file
quant-infer -m model.quant -p "Tell me a story" -o story.txt --max-tokens 200
```

---

### quant-train - Train a Model from Scratch

**Description:** Train a new model from scratch.

**Usage:**
```bash
quant-train [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config`, `-c` | string | required | Path to training configuration JSON file |
| `--data`, `-d` | string | required | Path to training data (text file) |
| `--output`, `-o` | string | required | Path to save the trained model (.quant) |
| `--batch-size`, `-b` | int | 4 | Batch size for training |
| `--seq-length`, `-s` | int | 128 | Sequence length for training |
| `--epochs`, `-e` | int | 1 | Number of training epochs |
| `--learning-rate`, `-lr` | float | 0.001 | Learning rate |
| `--optimizer` | string | adamw | Optimizer: sgd, adam, adamw |
| `--weight-decay` | float | 0.01 | Weight decay for optimizer |
| `--warmup-steps` | int | 100 | Learning rate warmup steps |
| `--log-interval` | int | 10 | Log training metrics every N steps |
| `--save-interval` | int | 1000 | Save checkpoint every N steps |
| `--gpu` | flag | false | Use GPU acceleration |
| `--resume` | string | | Path to checkpoint to resume training |
| `--seed` | int | 42 | Random seed |
| `--verbose`, `-v` | flag | false | Verbose output |

**Configuration File Example (`config.json`):**

```json
{
  "model": {
    "type": "dense",
    "dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "vocab_size": 50257,
    "norm_eps": 1e-6
  },
  "format": {
    "target_bpw": 1.50
  },
  "training": {
    "batch_size": 4,
    "seq_length": 128,
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adamw"
  }
}
```

**Examples:**

```bash
# Train with configuration file
quant-train -c config.json -d data/tinyshakespeare.txt -o model.quant

# Train with command-line options
quant-train -d data.txt -o model.quant --dim 512 --n-layers 6 --epochs 5

# Train with GPU
quant-train -c config.json -d data.txt -o model.quant --gpu
```

---

### quant-finetune - Fine-tune an Existing Model

**Description:** Fine-tune a pre-trained model on new data.

**Usage:**
```bash
quant-finetune [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model`, `-m` | string | required | Path to base model (.quant) |
| `--data`, `-d` | string | required | Path to fine-tuning data (text file) |
| `--output`, `-o` | string | required | Path to save fine-tuned model (.quant) |
| `--epochs`, `-e` | int | 1 | Number of fine-tuning epochs |
| `--learning-rate`, `-lr` | float | 1e-5 | Learning rate (typically smaller than training) |
| `--batch-size`, `-b` | int | 4 | Batch size |
| `--seq-length`, `-s` | int | 128 | Sequence length |
| `--method` | string | full | Fine-tuning method: full, quantized |
| `--target-bpw` | float | 1.50 | Target bits-per-weight for quantized fine-tuning |
| `--target-modules` | string | all | Modules to fine-tune (comma-separated) |
| `--freeze-base` | flag | false | Freeze base model weights |
| `--log-interval` | int | 10 | Log every N steps |
| `--save-interval` | int | 100 | Save checkpoint every N steps |
| `--gpu` | flag | false | Use GPU acceleration |
| `--seed` | int | 42 | Random seed |
| `--verbose`, `-v` | flag | false | Verbose output |

**Examples:**

```bash
# Full fine-tuning
quant-finetune -m base.quant -d data.txt -o fine-tuned.quant --epochs 3

# Quantized fine-tuning (native QUANT, target 2.0 BPW)
quant-finetune -m base.quant -d data.txt -o quant.quant --method quantized --target-bpw 2.0
```

---

### quant-convert - Convert Model Formats

**Description:** Convert models between different formats.

**Usage:**
```bash
quant-convert [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input`, `-i` | string | required | Input model file |
| `--output`, `-o` | string | required | Output model file (.quant) |
| `--format` | string | rawfp32 | Input format: rawfp32, gguf |
| `--bpw` | float | 0 | Target bits per weight for FormatPlanner (0 = no compression) |
| `--codebook-size` | int | 256 | Codebook size for QUANT8 (256) or QUANT4 (16) |
| `--verbose`, `-v` | flag | false | Verbose output |

**Supported Input Formats:**
- raw FP32 binary
- `gguf` (GGML format)

**Examples:**

```bash
# Convert raw FP32 to QUANT
quant-convert -i model.rawfp32 -o model.quant --bpw 1.50

# Convert with different BPW
quant-convert -i model.rawfp32 -o model_small.quant --bpw 2.0

# Convert GGUF to QUANT
quant-convert -i model.gguf -o model.quant --format gguf
```

---

### quant-info - Display Model Information

**Description:** Display information about a Transcender model.

**Usage:**
```bash
quant-info [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model`, `-m` | string | required | Path to the model file (.quant) |
| `--json` | flag | false | Output in JSON format |
| `--verbose`, `-v` | flag | false | Verbose output |

**Output Includes:**
- Model architecture (type, dimensions, layers)
- Format information (BPW, formats used)
- Parameter count
- Memory usage
- Supported operations
- Metadata

**Examples:**

```bash
# Basic info
quant-info -m model.quant

# JSON output
quant-info -m model.quant --json > model_info.json

# Verbose output
quant-info -m model.quant -v
```

**Example Output:**

```
Model: model.quant
Type: Dense Transformer
Dimensions: 512
Layers: 6
Heads: 8
Vocab Size: 50257
Parameters: 12,345,678

Format: QUANT (Mixed, v3 Q-series — e.g. QG8/Q8 for salient, Q4 for mid, Q1/Q2 for bulk; NO Ternary/Binary — removed per pure-Q policy, ledger C-01)
Average BPW: 1.50 (UNVERIFIED example value — illustrative only)
Formats (UNVERIFIED example split — illustrative only):
  - Q8/QG8: salient weights
  - Q4: moderately important weights
  - Q1/Q2: bulk weights

Codebook:
  - QUANT8: 256 entries (FP32)
  - QUANT4: 16 entries (FP16)

Memory Usage:
  - Model: 45.2 MB
  - Codebooks: 1.1 MB
  - Total: 46.3 MB
```

---

### quant-bench - Run Performance Benchmarks

**Description:** Benchmark model performance.

**Usage:**
```bash
quant-bench [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model`, `-m` | string | required | Path to the model file (.quant) |
| `--prompts`, `-p` | string | required | Path to file with benchmark prompts |
| `--iterations`, `-n` | int | 100 | Number of iterations |
| `--warmup` | int | 10 | Number of warmup iterations |
| `--batch-size` | int | 1 | Batch size for benchmarking |
| `--max-tokens` | int | 256 | Maximum tokens to generate |
| `--gpu` | flag | false | Use GPU acceleration |
| `--csv` | flag | false | Output results in CSV format |
| `--output`, `-o` | string | stdout | Output file for results |
| `--verbose`, `-v` | flag | false | Verbose output |

**Prompts File Format:**

```text
# Comments start with #
Once upon a time
Write a poem about AI
The quick brown fox jumps over the lazy dog
Explain quantum computing to a 5-year-old
```

**Output Metrics:**
- Tokens per second
- Time per token (ms)
- Prefill time
- Generation time
- Memory usage

**Examples:**

```bash
# Basic benchmark
quant-bench -m model.quant -p bench/prompts.txt -n 100

# With warmup and CSV output
quant-bench -m model.quant -p prompts.txt -n 1000 --warmup 50 --csv -o results.csv

# GPU benchmark
quant-bench -m model.quant -p prompts.txt -n 100 --gpu
```

---

## 📖 Using the C++ API

For programmatic access to Transcender functionality, you can use the C++ API directly.

### Basic Setup

```cpp
#include <quant/quant.h>
#include <iostream>

using namespace quant;

int main() {
    // Initialize the QUANT engine
    BackendConfig config;
    config.device = Device::CPU;
    config.precision = Precision::FP32;
    
    Backend backend(config);
    
    // Load a model
    Model* model = Model::load("model.quant", backend);
    
    // Create a tokenizer
    Tokenizer tokenizer("tokenizer.json");
    
    // Tokenize input
    std::vector<int> tokens = tokenizer.encode("Hello, world!");
    
    // Run inference
    Tensor input(Shape{1, (int)tokens.size()}, DType::I64);
    std::copy(tokens.begin(), tokens.end(), input.data<int64_t>());
    
    Tensor output = model->forward(input);
    
    // Sample from output
    int next_token = sampler.sample(output);
    
    // Decode token
    std::string text = tokenizer.decode({next_token});
    
    std::cout << "Generated: " << text << std::endl;
    
    // Cleanup
    delete model;
    
    return 0;
}
```

### Training Example

```cpp
#include <quant/quant.h>
#include <iostream>

using namespace quant;

int main() {
    // Create a model
    TransformerConfig config;
    config.dim = 512;
    config.n_layers = 6;
    config.n_heads = 8;
    config.vocab_size = 50257;
    
    DenseModel model(config);
    
    // Create tokenizer
    Tokenizer tokenizer("tokenizer.json");
    
    // Load training data
    DataLoader dataloader(&tokenizer, "data.txt", 4, 128);
    
    // Create optimizer
    AdamW optimizer(0.001, 0.01);
    
    // Create trainer
    Trainer trainer(&model, &tokenizer);
    trainer.compile(&optimizer);
    
    // Training configuration
    TrainConfig train_config;
    train_config.batch_size = 4;
    train_config.seq_length = 128;
    train_config.num_epochs = 10;
    train_config.learning_rate = 0.001;
    train_config.log_interval = 10;
    train_config.save_interval = 100;
    train_config.output_path = "trained.quant";
    
    // Set logging callback
    trainer.set_log_callback([](const TrainMetrics& metrics) {
        std::cout << "Step: " << metrics.step
                  << " Loss: " << metrics.loss
                  << " PPL: " << metrics.perplexity
                  << std::endl;
    });
    
    // Train!
    trainer.fit(dataloader, train_config);
    
    return 0;
}
```

---

### Inference with GPU Acceleration

```cpp
#include <quant/quant.h>
#include <quant/gpu_compute.h>
#include <iostream>

using namespace quant;

int main() {
    // Initialize GPU backend
    BackendConfig config;
    config.device = Device::GPU;
    config.precision = Precision::FP16;
    
    Backend backend(config);
    
    // Load model
    Model* model = Model::load("model.quant", backend);
    
    // Tokenize
    Tokenizer tokenizer("tokenizer.json");
    std::vector<int> tokens = tokenizer.encode("Hello");
    
    // Create input tensor
    Tensor input(Shape{1, (int)tokens.size()}, DType::I64);
    std::copy(tokens.begin(), tokens.end(), input.data<int64_t>());
    
    // Run inference on GPU
    Tensor output = model->forward(input);
    
    // Sample
    Sampler sampler(0.7f, 50);
    int next_token = sampler.sample(output);
    
    std::cout << "Generated token: " << next_token << std::endl;
    
    delete model;
    return 0;
}
```

---

## 🔧 Configuration Files

Transcender uses JSON files for configuration.

### Model Configuration

```json
{
  "model": {
    "type": "dense",
    "dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "vocab_size": 50257,
    "norm_eps": 1e-6,
    "dropout": 0.1
  }
}
```

### Training Configuration

```json
{
  "training": {
    "batch_size": 4,
    "seq_length": 128,
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "log_interval": 10,
    "save_interval": 1000
  },
  "format": {
    "target_bpw": 1.50
  }
}
```

### Quantization Configuration

```json
{
  "quantization": {
    "target_bpw": 1.50,
    "calibration_data": "calib.txt",
    "codebook_size_quant8": 256,
    "codebook_size_quant4": 16,
    "salient_percentile": 0.01,
    "moderate_percentile": 0.04
  }
}
```

---

## 🎯 Best Practices

### 1. Model Selection

| Task | Recommended Model Size | Notes |
|------|----------------------|-------|
| Testing/Development | 10M-50M parameters | Fast to train and run |
| Small Applications | 50M-200M parameters | Good balance of quality and speed |
| Production | 200M-2B parameters | Best quality, requires GPU |
| Research | 2B+ parameters | Experimental, requires significant resources |

### 2. Quantization Settings

| Use Case | Target BPW | Formats Used |
|----------|-----------|--------------|
| Maximum Quality | 3.0+ | Mostly Q8/QG8 (UNVERIFIED guidance — no end-task bench source) |
| Balanced | 1.50-2.0 | Q8 + Q4 + Q1/Q2 mix (UNVERIFIED guidance) |
| Compact | 1.0-1.5 | Q4 + Q1/Q2 (UNVERIFIED guidance) |
| Minimum Size | < 1.0 | Q1 family (UNVERIFIED guidance — sub-1.0 has no registered format; minimum is Q1=1.0) |

### 3. Training Tips

- Start with a small model for testing
- Use a learning rate finder to determine optimal LR
- Warmup is important for Adam optimizers
- Monitor loss and perplexity
- Save checkpoints regularly

### 4. Inference Tips

- Use `--max-tokens` to limit generation length
- Adjust `--temperature` for creativity vs. coherence
- Use `--top-k` and `--top-p` together for best results
- Lower temperature = more deterministic
- Higher temperature = more creative/random

---

## 🐛 Common Issues

### 1. Out of Memory

**Solution:**
- Reduce batch size
- Reduce sequence length
- Use smaller model
- Enable quantization to reduce memory usage
- Use CPU instead of GPU (if applicable)

### 2. Slow Inference

**Solution:**
- Enable AVX2 optimizations (`-DQUANT_AVX2=ON`)
- Use GPU acceleration (`--gpu`)
- Quantize the model to lower BPW
- Reduce batch size

### 3. Model Fails to Load

**Solution:**
- Check file path is correct
- Verify model format is supported
- Check for file corruption
- Ensure all required files are present

### 4. Training Loss Doesn't Decrease

**Solution:**
- Check learning rate (try 1e-4 to 1e-3)
- Verify data is properly tokenized
- Increase batch size or sequence length
- Try different optimizer
- Monitor gradients for vanishing/exploding

### 5. Poor Quality Output

**Solution:**
- Increase model size
- Train for more epochs
- Use higher quality training data
- Adjust quantization settings
- Try different sampling parameters

---

## 📚 Additional Resources

- **[Wiki Usage Guide](../wiki/Usage-Guide.md)** — Extended usage examples and tips
- **[Wiki Inference Guide](../wiki/Inference.md)** — In-depth inference documentation
- **[Wiki Training Guide](../wiki/Training.md)** — Detailed training instructions
- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Architecture](ARCHITECTURE.md)** - System architecture overview
- **[Research](RESEARCH.md)** - Research papers and algorithms
- **[Examples](EXAMPLES/)** - Code examples and tutorials

---

*Last updated: July 12, 2026*
