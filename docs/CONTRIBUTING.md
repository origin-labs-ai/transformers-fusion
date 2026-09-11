# Contributing to Transcender

> **How to Contribute to the Project**
>
> **Production-hardening note (2026-09-11):** v1.1.0 / R0001.01 · 105 formats · TranscenderIDX · 72 ctest cases
> (`tests/CMakeLists.txt`). Source truth: `src/<area>/` layout
> (`docs/ARCHITECTURE.md`); open tasks: `docs/GOOD_FIRST_ISSUES.md`;
> measurement rule: every number ships with a bench/ledger source or is flagged
> UNVERIFIED.

---

## 🎉 Welcome!

Thank you for your interest in contributing to Transcender! We welcome contributions from everyone, whether you're a seasoned developer or just starting out with AI and C++.

This document provides guidelines for contributing to Transcender. Following these guidelines helps maintain a consistent, high-quality codebase and makes it easier for everyone to collaborate.

---

## 📋 Ways to Contribute

There are many ways to contribute to Transcender:

### 🎯 Find Your First Issue (Start Here!)

Don't know where to begin? We maintain a **live contributor board** with every
open task tiered by difficulty — starter, intermediate, and advanced:

**👉 [docs/GOOD_FIRST_ISSUES.md](GOOD_FIRST_ISSUES.md)**

Pick an ID (e.g. `GF-01`, `INT-05`, `ADV-03`), comment "taking <ID>" on its
issue, and you own it. Each entry lists exact files and exit criteria.

### 🐛 Bug Reports

- **Find a bug?** Open an issue with a clear description
- **Have a fix?** Submit a pull request
- **Confirm a bug?** Comment on existing issues with additional information

### 💡 Feature Requests

- **Have an idea?** Open an issue to discuss it
- **Want to implement?** Submit a pull request
- **Need clarification?** Ask questions in discussions

### 📖 Documentation

- **Find a typo?** Fix it!
- **Missing docs?** Add them!
- **Confusing explanation?** Improve it!

### 🧪 Testing

- **Find a test gap?** Add a test!
- **Improve coverage?** Help us reach 100%
- **Performance testing?** Benchmark and optimize

### 🚀 Development

- **Fix bugs** in existing code
- **Implement new features** from the roadmap
- **Optimize** existing implementations
- **Refactor** for better maintainability

### 🎓 Research

- **Implement** new research papers
- **Propose** new algorithms
- **Benchmark** against other frameworks
- **Publish** research using Transcender

### 🤝 Community

- **Answer questions** from other users
- **Review** pull requests
- **Help** with onboarding
- **Promote** Transcender

---

## 🏁 Getting Started

### 1. Set Up Your Development Environment

Follow the [Build Guide](BUILD.md) to get Transcender compiled on your machine:

```bash
# Clone the repository
git clone https://github.com/origin-labs-ai/Transcender
cd Transcender

# Build in debug mode for development
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

### 2. Run the Tests

Make sure everything works:

```bash
ctest --test-dir build --output-on-failure -j$(nproc)
```

### 3. Create a Branch

Create a feature branch for your changes:

```bash
git checkout -b feature/your-feature-name
git checkout -b fix/your-bug-fix
git checkout -b doc/your-documentation
```

Branch naming conventions:
- `feature/...` - New features
- `fix/...` - Bug fixes
- `doc/...` - Documentation improvements
- `refactor/...` - Code refactoring
- `perf/...` - Performance optimizations
- `test/...` - Test improvements

---

## 📝 Pull Request Guidelines

### Before Submitting

1. **Check for existing issues** - Someone might already be working on it
2. **Discuss large changes** - Open an issue first for significant changes
3. **Run tests** - Make sure all existing tests pass
4. **Add tests** - Add tests for new functionality
5. **Update docs** - Update documentation if needed
6. **Format code** - Follow the code style guidelines

### Pull Request Checklist

- [ ] Code compiles without warnings
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated (if applicable)
- [ ] Code follows style guidelines
- [ ] Commit messages are clear and descriptive
- [ ] Changes are focused on a single purpose

### Pull Request Template

```markdown
## Description

[Clear description of what this PR does]

## Related Issues

[Link to related issues or discussions]

## Changes

- [ ] New feature
- [ ] Bug fix
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Test coverage
- [ ] Refactoring

## Testing

[Describe how you tested your changes]

## Notes

[Any additional information, limitations, or future work]
```

---

## 📛 Code Style Guidelines

### General Principles

1. **Readability First** - Code should be easy to read and understand
2. **Consistency** - Follow existing patterns in the codebase
3. **Minimalism** - Keep code simple and focused
4. **Performance** - Optimize when necessary, but don't sacrifice readability

### C++ Style

#### Formatting

- **Indentation:** 4 spaces (no tabs)
- **Braces:** Opening brace on same line, closing on new line
- **Line length:** Maximum 100 characters (soft limit)
- **Spacing:** Use spaces around operators, after commas
- **Comments:** Use `//` for single-line, `/* */` for multi-line

```cpp
// Good
void function(int a, int b) {
    int c = a + b;
    return c * 2;
}

// Bad
void function(int a,int b){int c=a+b;return c*2;}
```

#### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes/Structs | PascalCase | `Transformer`, `Tensor` |
| Functions | snake_case | `forward()`, `save_model()` |
| Variables | snake_case | `input_ids`, `weight_matrix` |
| Constants | UPPER_SNAKE_CASE | `MAX_TOKENS`, `DEFAULT_DIM` |
| Member variables | snake_case with underscore suffix | `weight_`, `dim_` |
| Namespace | lowercase | `quant`, `quant::math` |
| Templates | PascalCase for template parameters | `T`, `InputType` |

```cpp
class Tensor {
private:
    Shape shape_;
    DType dtype_;
    void* data_;
public:
    Shape shape() const { return shape_; }
};
```

#### Header Files

- Use `#pragma once` instead of include guards
- Group includes logically
- Forward declare when possible
- Use inline comments for brief documentation

```cpp
#pragma once

#include <cstdint>
#include <vector>
#include <string>

// Other QUANT headers
#include "quant/tensor.h"
#include "quant/types.h"

namespace quant {

class Tensor;

class Shape {
    // ...
};

} // namespace quant
```

#### Classes

- Use `public:` for interface, `private:` for implementation
- Declare member functions and variables separately
- Use `const` for methods that don't modify state
- Use `override` for virtual function overrides
- Use `final` for classes that shouldn't be inherited

```cpp
class Tensor final {
public:
    Tensor(const Shape& shape, DType dtype);
    
    const Shape& shape() const;
    DType dtype() const;
    
    void fill_(float value);
    
private:
    Shape shape_;
    DType dtype_;
    void* data_;
};
```

#### Functions

- Keep functions short (preferably < 50 lines)
- Use descriptive names
- Pass by const reference for large objects
- Return by value (RVO/NRVO will handle it)
- Use default arguments judiciously

```cpp
// Good
Tensor matmul(const Tensor& a, const Tensor& b);
void save_to_file(const Model& model, const std::string& path);

// Bad
Tensor m(const Tensor& a, const Tensor& b);  // Unclear name
void save(const Model& m, std::string p);    // No const, pass by value
```

### Comments

#### When to Comment

- **Why:** Explain the reason behind non-obvious code
- **What:** Explain complex algorithms or data structures
- **How:** Document non-standard usage or patterns
- **TODO:** Mark incomplete work or future improvements

#### When NOT to Comment

- Obvious code (e.g., `x = x + 1; // Increment x`)
- Standard library usage
- Simple getters/setters

#### Comment Style

```cpp
// Single line comments for brief explanations

/*
 * Multi-line comments for longer explanations
 * or documentation
 */

/// Doxygen-style comments for API documentation
/// @param x Input tensor
/// @return Result tensor
Tensor function(const Tensor& x);
```

### Error Handling

- Use `QUANT_CHECK` for precondition checks
- Use exceptions sparingly (only for unrecoverable errors)
- Prefer returning error codes or `std::optional` for recoverable errors
- Include context in error messages

```cpp
// Good
QUANT_CHECK(shape.rank() == 2, "Expected 2D tensor, got " + std::to_string(shape.rank()));

// Bad
if (shape.rank() != 2) {
    throw std::runtime_error("Bad shape");  // Not descriptive
}
```

---

## 🗂️ Code Organization

### File Structure

```
Transcender/
├── CMakeLists.txt              # Main build configuration
├── README.md                   # Project documentation
├── include/                    # Public headers
│   └── quant/                    # All QUANT headers
│       ├── types.h            # Core types
│       ├── tensor.h           # Tensor class
│       ├── autograd.h         # Autograd engine
│       └── ...
├── src/                       # Source files
│   ├── tensor.cpp             # Tensor implementation
│   ├── autograd.cpp           # Autograd implementation
│   └── ...
├── engines/                    # Engines (inference, trainer)
│   ├── inference/             # Inference engine
│   └── trainer/               # Training engine
├── tools/                      # CLI tools
│   ├── convert.cpp            # Model conversion
│   └── ...
└── tests/                     # Tests
    ├── test_tensor.cpp        # Tensor tests
    └── ...
```

### Header File Guidelines

- Public headers go in `include/quant/`
- Each `.cpp` file should have a corresponding header
- Headers should be self-contained (include all dependencies)
- Use forward declarations to minimize includes

---

## 🧪 Testing Guidelines

### Test Structure

```cpp
#include "quant/quant.h"
#include <catch2/catch.hpp>  // Or use Google Test

namespace quant {

TEST_CASE("Tensor operations", "[tensor]") {
    SECTION("Construction") {
        Tensor t({2, 3}, DType::F32);
        REQUIRE(t.shape().dims == std::vector<int64_t>{2, 3});
    }
    
    SECTION("Addition") {
        Tensor a = Tensor::ones({2, 3});
        Tensor b = Tensor::ones({2, 3});
        Tensor c = a + b;
        // ... assertions
    }
}

} // namespace quant
```

### Test Guidelines

1. **Test one thing per test** - Keep tests focused
2. **Use descriptive names** - `test_tensor_addition` not `test_1`
3. **Test edge cases** - Empty tensors, boundary values, etc.
4. **Test error conditions** - Invalid inputs, out-of-bounds, etc.
5. **Keep tests fast** - Avoid slow operations in tests
6. **Use helper functions** - For common test patterns

### Test Coverage

Aim for 100% coverage of:
- All public APIs
- All branches and conditions
- All error paths

Check coverage with:
```bash
# Using gcov (Linux)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug --enable-coverage
cmake --build build --parallel
ctest --test-dir build --output-on-failure -j$(nproc)

# Generate coverage report
gcovr -r . --html --html-details -o coverage.html
```

---

## 📝 Commit Guidelines

### Commit Messages

Use clear, descriptive commit messages following this format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `perf` - Performance improvement
- `refactor` - Code refactoring
- `doc` - Documentation changes
- `test` - Test changes
- `build` - Build system changes
- `ci` - CI/CD changes
- `chore` - Other changes

**Examples:**

```
feat(tensor): Add view() method for zero-copy slicing

- Add view() method to Tensor class
- Add corresponding tests
- Update documentation

fix(autograd): Fix gradient accumulation for bias parameters

- Fix bug in bias_add_op gradient computation
- Add regression test
- Closes #123

perf(math): Optimize AVX2 GEMM with better tiling

- Improve cache locality in GEMM kernel
- Benchmark shows 15% speedup on Zen 3
```

### Commit Size

- **Small commits** - Each commit should do one logical thing
- **Atomic changes** - Each commit should be self-contained
- **Frequent commits** - Commit often to make review easier

### Git History

- **Don't rewrite public history** - No force-pushing to main
- **Rebase private branches** - Keep your feature branch clean
- **Squash merge** - For small PRs, squash into one commit
- **Merge commit** - For large PRs, keep the history

---

## 👥 Code Review Process

### For Contributors

1. **Submit PR** - Create a pull request with clear description
2. **Address feedback** - Respond to review comments promptly
3. **Update PR** - Push changes to address feedback
4. **Wait for approval** - At least one maintainer must approve
5. **Merge** - Once approved, a maintainer will merge

### For Reviewers

1. **Be constructive** - Focus on improving the code
2. **Be timely** - Review promptly if you can
3. **Be specific** - Point to exact lines and explain issues
4. **Suggest alternatives** - Don't just say "this is wrong"
5. **Approve when ready** - Don't nitpick indefinitely

### Review Checklist

- [ ] Code compiles and tests pass
- [ ] Follows code style guidelines
- [ ] Proper error handling
- [ ] Good variable/function names
- [ ] Appropriate comments
- [ ] No unnecessary complexity
- [ ] Performance considerations
- [ ] Documentation updated

---

## 📊 Performance Guidelines

### When to Optimize

1. **Profile first** - Identify bottlenecks with profiling
2. **Measure** - Quantify improvements
3. **Focus on hot paths** - Optimize code that runs frequently
4. **Balance readability** - Don't sacrifice clarity for micro-optimizations

### Optimization Techniques

```cpp
// Before optimizing
void slow_function(Tensor& a, Tensor& b, Tensor& out) {
    for (int i = 0; i < a.shape()[0]; i++) {
        for (int j = 0; j < a.shape()[1]; j++) {
            out(i, j) = a(i, j) + b(i, j);
        }
    }
}

// After optimizing (with AVX2)
void fast_function(Tensor& a, Tensor& b, Tensor& out) {
    // Check alignment
    // Use SIMD instructions
    // Optimize memory access patterns
}
```

### Profiling Tools

```bash
# Linux: perf
perf stat -e cache-misses,cycles ./your_program

# Linux: gprof
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPROFILE=ON
cmake --build build --parallel
./build/bin/your_program
gprof build/bin/your_program gmon.out > analysis.txt

# Windows: Visual Studio profiler
# macOS: Instruments
```

---

## 📚 Documentation Guidelines

### Documentation Standards

1. **API Documentation** - Use Doxygen-style comments for all public APIs
2. **Code Comments** - Explain why, not what (the code should be self-explanatory)
3. **README Updates** - Keep the main README up-to-date
4. **Change Log** - Document breaking changes

### Doxygen Example

```cpp
/// @brief Multiplies two tensors
/// @param a First input tensor (M x K)
/// @param b Second input tensor (K x N)
/// @return Result tensor (M x N)
/// @throws std::invalid_argument If dimensions don't match
Tensor matmul(const Tensor& a, const Tensor& b);
```

---

## 🎯 Roadmap & Priorities

### Current Priorities (v0.2)

1. **Vision Module** - Complete vision transformer implementation
2. **Improved MoE** - Better routing, more variants
3. **More Tools** - Additional CLI utilities
4. **Performance** - Optimize critical paths
5. **Documentation** - Complete API documentation

### Future Roadmap

| Version | Focus | Target Date |
|---------|-------|-------------|
| v0.2 | Vision + MoE improvements | Q3 2026 |
| v0.3 | Distributed training | Q4 2026 |
| v0.4 | Multi-modal support | Q1 2027 |
| v1.0 | Production-ready | 2027 |

### Good First Issues

Look for issues labeled with:
- `good first issue` - Beginner-friendly
- `help wanted` - Needs contribution
- `documentation` - Documentation tasks
- `tests` - Test improvements

---

## 🤝 Community Guidelines

### Code of Conduct

Be respectful and inclusive. Follow the [Contributor Covenant](https://www.contributor-covenant.org/).

### Communication

- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions and discussions
- **Pull Requests** - For code contributions
- **Email** - For private matters (if necessary)

### Recognition

All contributors will be recognized:
- In the CONTRIBUTORS file
- In the release notes
- In the GitHub contributors graph

---

## 📄 License

By contributing to Transcender, you agree that your contributions will be licensed under the **MIT License**. See [LICENSE](../LICENSE) for details.

---

## 🎉 Thank You!

Your contributions help make Transcender better for everyone. Thank you for being part of the community!

---

## 📞 Need Help?

- **New to Git?** Check out [GitHub's Git Handbook](https://guides.github.com/introduction/git-handbook/)
- **New to C++?** Check out [learncpp.com](https://www.learncpp.com/)
- **New to AI?** Check out [fast.ai](https://course.fast.ai/)
- **Questions?** Open a discussion or ask in a PR comment
- **Per-file docs?** See the [wiki/files/](../wiki/files/_index.md) for detailed source documentation

---

*Last updated: July 12, 2026*
