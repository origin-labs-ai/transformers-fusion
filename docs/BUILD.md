# Build & Installation Guide

> **Compiling Transcender from Source**
>
> **Production-hardening sync (2026-09-11):** version one-truth **1.1.0 / R0001.01**
> (`CMakeLists.txt:3` `project(Transcender VERSION 1.1.0)`,
> `include/quant/version.h:6` `Transcender_VERSION_STRING "R0001.01"` — the "0.2.0"
> in older Phase 24 banners is STALE). Build graph one-truth: **26 libs +
> 14 tools + 12 benches + SOPS/GLE + 72 ctest cases ≈ 111 materialized targets**
> (ledger C-08 VERIFIED). Test count "42"/"62" anywhere is STALE — truth is 72 ctest cases
> from 73 test files (`ctest -N` = 72; 71× `add_quant_test_full` + `test_gpu`);
> **72/72 green on 2026-09-11** (Release, `ctest --test-dir build -C Release),
> evidence `build/Testing/Temporary/LastTest.log` (ledger C-07 VERIFIED this round).

---

## 📋 Quick Start

### Windows (Clang-cl)
```powershell
# Clone the repository
git clone https://github.com/origin-labs-ai/transformers-fusion
cd transformers-fusion

# Configure (requires CMake >= 3.24, Ninja recommended)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build everything (libraries + tools + tests + benchmarks)
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure
```

### Linux (GCC/Clang)
```bash
# Clone the repository
git clone https://github.com/origin-labs-ai/transformers-fusion
cd transformers-fusion

# Install dependencies (CMake, Ninja, compiler)
# Ubuntu/Debian:
sudo apt-get install cmake ninja-build g++

# Configure
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure -j$(nproc)
```

### macOS (Apple Clang)
```bash
# Clone the repository
git clone https://github.com/origin-labs-ai/transformers-fusion
cd transformers-fusion

# Install CMake and Ninja
brew install cmake ninja

# Configure
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure -j$(sysctl -n hw.ncpu)
```

---

## 📦 Prerequisites

### Required Dependencies

| Dependency | Minimum Version | Purpose | Notes |
|-----------|----------------|---------|-------|
| **CMake** | 3.24 | Build system | Required for all platforms |
| **C++20 Compiler** | Varies | C++20 support | See below for platform-specific requirements |
| **Ninja** | 1.11 | Build system generator | Optional but recommended |

### Compiler Requirements

| Platform | Compiler | Minimum Version | Notes |
|----------|----------|----------------|-------|
| Windows | Clang-cl | 16 | Recommended for best compatibility |
| Windows | MSVC | 2022 (17.0) | Works but Clang-cl recommended |
| Linux | GCC | 12 | Full C++20 support |
| Linux | Clang | 16 | Full C++20 support |
| macOS | Apple Clang | 14 (Xcode 13+) | C++20 support may vary |

### Optional Dependencies

| Dependency | Purpose | Status |
|-----------|---------|--------|
| **DirectX 12 SDK** | GPU acceleration | Optional - Required for GPU support |
| **Python** | None - All tooling is C++ | Not required |

---

## 🛠️ Build Options

Transcender uses CMake options to configure the build. You can enable/disable features using `-D` flags.

### Standard Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug, Release, RelWithDebInfo) |
| `QUANT_BUILD_TESTS` | ON | Build unit tests |
| `QUANT_BUILD_BENCHMARKS` | ON | Build benchmark executables |
| `QUANT_BUILD_TOOLS` | ON | Build CLI tools |

### Platform-Specific Options

| Option | Default | Description |
|--------|---------|-------------|
| `QUANT_AVX2` | ON (if available) | Enable AVX2 SIMD optimizations |
| `QUANT_USE_DIRECTX` | OFF | Enable DirectX 12 GPU support |
| `QUANT_SANITIZE` | OFF | Enable address sanitizer |

---

## 🎯 Build Configurations

### Release Build (Recommended)

For production use, maximum performance:

```bash
# Windows
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Linux/macOS
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```

**Characteristics:**
- Optimizations enabled (`-O3`)
- Debug symbols stripped
- No runtime checks
- Fastest execution

---

### Debug Build

For development and debugging:

```bash
# Windows
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel

# Linux/macOS
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel $(nproc)
```

**Characteristics:**
- No optimizations (`-O0`)
- Debug symbols included
- Runtime checks enabled
- Easier debugging
- Slower execution

---

### Debug with Sanitizers

For memory error detection:

```bash
# Address Sanitizer
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DQUANT_SANITIZE=ON
cmake --build build --parallel

# Run with sanitizer
./build/tests/test_tensor
```

**Note:** Sanitizers add significant overhead and may slow down execution by 2-10x.

---

### AVX2 Optimized Build

To explicitly enable AVX2 optimizations:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DQUANT_AVX2=ON
cmake --build build --parallel
```

**Note:** AVX2 is automatically detected and enabled if your CPU supports it.

---

### GPU-Enabled Build (Windows Only)

To enable DirectX 12 GPU support:

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DQUANT_USE_DIRECTX=ON
cmake --build build --parallel
```

**Prerequisites:**
- DirectX 12 SDK installed
- Windows 10/11
- Supported GPU

---

### Minimal Build (Core Only)

If you only want the core library without tools, tests, or benchmarks:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DQUANT_BUILD_TESTS=OFF \
    -DQUANT_BUILD_BENCHMARKS=OFF \
    -DQUANT_BUILD_TOOLS=OFF
cmake --build build --parallel
```

---

### Custom Install Prefix

To install to a custom location:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build build --parallel
cmake --install build
```

---

## 📁 Build Output Structure

After building, your `build/` directory will contain:

```
build/
├── bin/                    # Tools and executables
│   ├── quant-convert         # Model format conversion
│   ├── quant-train           # Training tool
│   ├── quant-infer           # Inference tool
│   ├── quant-finetune        # Fine-tuning tool
│   ├── quant-info            # Model information
│   ├── quant-bench           # Benchmarking
│   └── tests/              # Test executables
│       ├── test_tensor
│       ├── test_math
│       ├── test_kernel
│       ├── test_model
│       ├── test_tokenizer
│       ├── test_format
│       ├── test_trainer
│       └── test_all
│
├── lib/                    # Libraries
│   ├── libquant_core.a       # Core library (static)
│   ├── libquant_math.a       # Math library
│   ├── libquant_format.a     # QUANT format library
│   └── ...
│
├── include/                # Public headers (symlink to source)
│   └── quant/                # All public QUANT headers
│
└── CMakeFiles/             # CMake build files
```

---

## 🏗️ Build System Details

### CMake Targets

| Target | Description | Dependencies |
|--------|-------------|--------------|
| `quant_core` | Core tensor and memory operations | None |
| `quant_math` | Mathematical operations | `quant_core` |
| `quant_format` | QUANT format reading/writing | `quant_math` |
| `quant_model` | Model implementations | `quant_format` |
| `quant_autograd` | Automatic differentiation | `quant_math` |
| `quant_trainer` | Training infrastructure | `quant_model`, `quant_autograd` |
| `quant_inference` | Inference engine | `quant_model` |
| `quant_gpu` | GPU acceleration | `quant_math` (Windows only) |

### CMake Custom Targets

| Target | Description |
|--------|-------------|
| `all` | Build all libraries and tools |
| `tests` | Build all test executables |
| `benchmarks` | Build all benchmark executables |
| `tools` | Build all CLI tools |

---

## 🔍 Build Customization

### Custom Compiler Flags

To add custom compiler flags:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic"
cmake --build build --parallel
```

### Custom Linker Flags

To add custom linker flags:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS="-static"
cmake --build build --parallel
```

---

## ⚡ Parallel Build

Transcender supports parallel builds for faster compilation:

```bash
# Use all available cores
cmake --build build --parallel

# Use specific number of cores
cmake --build build --parallel 8

# Ninja-specific
ninja -C build -j$(nproc)
```

**Note:** The first build will be slow as it compiles everything. Subsequent builds are much faster due to incremental compilation.

---

## 🧹 Clean Build

To clean and rebuild from scratch:

```bash
# Remove build directory
rm -rf build/

# Or use CMake to clean
cmake --build build --clean-first

# Reconfigure and rebuild
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CMake version too old

**Error:** `CMake 3.24 or higher is required`

**Solution:** Upgrade CMake
```bash
# Linux
sudo apt-get upgrade cmake

# macOS
brew upgrade cmake

# Windows
Download latest from https://cmake.org/download/
```

#### 2. Compiler doesn't support C++20

**Error:** `C++20 standard is not supported`

**Solution:** Upgrade your compiler or use a different one:
```bash
# Use Clang instead of GCC
sudo apt-get install clang
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang

# Use newer GCC
sudo apt-get install g++-12
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=g++-12
```

#### 3. Missing Ninja

**Error:** `Ninja not found`

**Solution:** Install Ninja
```bash
# Linux
sudo apt-get install ninja-build

# macOS
brew install ninja

# Windows (via Chocolatey)
choco install ninja
```

#### 4. DirectX 12 SDK not found

**Error:** `DirectX 12 SDK not found`

**Solution:** Install DirectX 12 SDK
- Download from: https://www.microsoft.com/en-us/download/details.aspx?id=6812
- Or install Windows 10/11 SDK which includes it

#### 5. Build fails with AVX2 errors

**Error:** `AVX2 instructions not supported`

**Solution:** Disable AVX2
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DQUANT_AVX2=OFF
cmake --build build --parallel
```

#### 6. Linker errors on Windows

**Error:** `LINK : fatal error LNK1120`

**Solution:** Make sure you're using the same compiler for both CMake configuration and building. Use Clang-cl consistently:
```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -T "ClangCL"
cmake --build build --parallel
```

---

## 📊 Build Performance

### Build Times (Approximate)

| Configuration | Machine | Time |
|--------------|---------|------|
| Release, no parallel | Laptop (4 cores) | 2-4 minutes |
| Release, parallel (8) | Desktop (8 cores) | 30-60 seconds |
| Release, parallel (16) | Workstation (16 cores) | 15-30 seconds |
| Debug, no parallel | Laptop (4 cores) | 3-6 minutes |
| Debug, parallel (8) | Desktop (8 cores) | 1-2 minutes |

### Build Size

| Configuration | Disk Space |
|--------------|------------|
| Release | ~200-400 MB |
| Debug | ~500-800 MB |
| With tests & tools | ~600-1000 MB |

---

## 🔄 Continuous Integration

For CI/CD pipelines, use:

```yaml
# GitHub Actions example
- name: Build and Test
  run: |
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel
    ctest --test-dir build --output-on-failure -j$(nproc)
```

---

## 📚 Additional Resources

- [Wiki Build Guide](../wiki/Build-Guide.md) — Extended build documentation with troubleshooting
- [CMake Documentation](https://cmake.org/documentation/)
- [Ninja Build System](https://ninja-build.org/)
- [Clang Documentation](https://clang.llvm.org/docs/)
- [GCC Documentation](https://gcc.gnu.org/onlinedocs/)

---

## 🎉 Success!

Once your build completes successfully, you're ready to:

1. **[Run tests](TESTING.md)** - Verify everything works
2. **[Use the tools](USAGE.md)** - Start working with models
3. **[Explore the API](API_REFERENCE.md)** - Build your own applications
4. **[Contribute](CONTRIBUTING.md)** - Help improve Transcender

---

*Last updated: July 26, 2026*
