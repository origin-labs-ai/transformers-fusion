# Internal Design Documents

> **Design Decisions, Rationale, and Technical Details**

---

## 📚 Overview

This directory contains **internal design documents** that explain the technical decisions behind Transcender's architecture. These documents are intended for:

- **Contributors** who need to understand the "why" behind design choices
- **Maintainers** who need to understand the system deeply
- **Architects** who are designing similar systems

Unlike the API documentation (which explains *how* to use the system) or the module documentation (which explains *what* each component does), these documents explain **why** things are designed the way they are.

---

## 🗂️ Available Documents

| Document | Description | Status |
|----------|-------------|--------|
| **[Format Design](format_design.md)** | Design of the QUANT binary format | ⏳ Planned |
| **[Memory Management](memory_management.md)** | Memory allocation strategies | ⏳ Planned |
| **[Kernel Optimization](kernel_optimization.md)** | SIMD and performance optimization strategies | ⏳ Planned |
| **[Autograd Design](autograd_design.md)** | Design of the automatic differentiation system | ⏳ Planned |
| **[Quantization Strategy](quantization_strategy.md)** | Overall quantization approach | ⏳ Planned |

---

## 🎯 Purpose of Internal Documentation

### Why We Need This

1. **Knowledge Preservation** - Capture decisions that might not be obvious from the code
2. **Onboarding** - Help new contributors understand the system quickly
3. **Consistency** - Ensure future changes align with the original vision
4. **Debugging** - Understand the rationale when investigating issues
5. **Refactoring** - Know what constraints exist when making changes

### What Belongs Here

✅ **Design Rationales** - Why we chose approach X over Y
✅ **Trade-offs** - What we sacrificed and why
✅ **Constraints** - Limitations and workarounds
✅ **Alternatives Considered** - What we tried and rejected
✅ **Future Directions** - Planned improvements and their rationale

❌ **API Documentation** - Belongs in [API_REFERENCE.md](../API_REFERENCE.md)
❌ **Usage Examples** - Belongs in [USAGE.md](../USAGE.md) or [EXAMPLES](../EXAMPLES/)
❌ **Module Overviews** - Belongs in [MODULES](../MODULES/)

---

## 📝 Document Templates

### Design Decision Document

```markdown
# [Feature/Component Name] - Design Document

## Status
- **Author:** [Name]
- **Date:** [Date]
- **Version:** [Version]
- **Status:** Draft / In Review / Approved / Deprecated

## Summary
[1-2 sentence summary of the decision]

## Background
[Context: What problem are we solving? Why is this important?]

## Requirements
[What must this solution achieve? What are the constraints?]

## Alternatives Considered

### Option 1: [Name]
**Pros:**
- [Pro 1]
- [Pro 2]

**Cons:**
- [Con 1]
- [Con 2]

**Decision:** [Accepted/Rejected] - [Reason]

### Option 2: [Name]
... [Same structure]

## Chosen Solution
[Detailed description of the chosen approach]

## Implementation Details
[How it's implemented, any non-obvious aspects]

## Trade-offs
[What we gained vs. what we lost]

## Future Improvements
[What could be improved later]

## Related Documents
- [Link to related design docs]
- [Link to relevant issues/PRs]
```

### Technical Deep Dive

```markdown
# [Topic] - Technical Deep Dive

## Overview
[What this document covers]

## The Problem
[Detailed explanation of the problem being solved]

## The Solution
[Detailed explanation of the solution]

## Mathematical Background
[Any relevant math, formulas, proofs]

## Implementation Details
[Code-level details, algorithms, data structures]

## Performance Analysis
[Complexity analysis, benchmarks, optimizations]

## Examples
[Concrete examples illustrating the concept]

## References
[Links to papers, articles, or other resources]
```

---

## 🎨 Design Principles

The following principles guide Transcender's design:

### 1. Zero Dependencies

**Principle:** Transcender should have no external dependencies.

**Rationale:**
- Eliminates version conflicts and dependency hell
- Reduces deployment complexity
- Enables air-gapped training
- Provides complete control over the codebase
- Makes the system more portable

**Trade-offs:**
- More code to maintain
- Must implement everything from scratch
- May be less optimized than specialized libraries

**Exceptions:**
- C++ standard library (required)
- CMake (build system, but users can use alternatives)
- DirectX 12 SDK (optional, for GPU support)

### 2. Single Format Truth

**Principle:** The `.quant` format should be the single source of truth for models.

**Rationale:**
- Eliminates format conversion overhead
- Ensures consistency across training and inference
- Simplifies the ecosystem
- Reduces complexity for users

**Trade-offs:**
- Must support all features in a single format
- Format must be extensible
- Conversion from other formats is required initially

### 3. Research-Driven Design

**Principle:** Every major design decision should be backed by research.

**Rationale:**
- Ensures we're using proven techniques
- Provides theoretical guarantees
- Avoids reinventing the wheel
- Makes the system more reliable

**Trade-offs:**
- Slower to adopt unproven techniques
- May miss out on cutting-edge innovations
- Requires significant literature review

### 4. Performance-First

**Principle:** Performance should be a primary consideration in all designs.

**Rationale:**
- AI workloads are computationally intensive
- Every optimization matters at scale
- Users expect fast performance
- Enables deployment on resource-constrained devices

**Trade-offs:**
- May increase code complexity
- May reduce readability
- May limit portability

### 5. Modularity

**Principle:** Components should be cleanly separated and loosely coupled.

**Rationale:**
- Easier to maintain
- Easier to test
- Easier to extend
- Easier for others to contribute
- Enables selective use of components

**Trade-offs:**
- May introduce some overhead
- May require more boilerplate
- May make some optimizations harder

---

## 🔍 Key Design Decisions

### Why C++20?

**Decision:** Use C++20 as the base standard.

**Rationale:**
- Modern C++ provides better abstractions
- C++20 has good compiler support
- Enables use of modern features (concepts, ranges, etc.)
- Better performance than C++17 and earlier
- Still widely supported

**Alternatives Considered:**
- **C++17:** More widely supported, but misses useful features
- **C++23:** More features, but less compiler support
- **C:** Simpler, but lacks OOP and templates
- **Rust:** Better safety, but steeper learning curve
- **Python:** Easier to use, but slower and has dependencies

### Why Not Use PyTorch/TensorFlow?

**Decision:** Build from scratch instead of using existing frameworks.

**Rationale:**
- **Zero dependencies:** PyTorch/TensorFlow have many dependencies
- **Control:** We can optimize for our specific use cases
- **Learning:** Building from scratch helps us understand deeply
- **Innovation:** Enables novel approaches not possible with existing frameworks
- **Format:** We can design the perfect format for our needs

**Trade-offs:**
- Significant development effort
- Must reimplement many features
- May have bugs that existing frameworks have already fixed
- Smaller community

### Why the QUANT Format?

**Decision:** Create a custom binary format (QUANT) instead of using existing formats.

**Rationale:**
- **Mixed formats:** Existing formats don't support per-block mixed quantization
- **Train-in-format:** Enables training directly in the compressed format
- **Codebooks:** Supports vector quantization with learned codebooks
- **Flexibility:** Can be extended for future needs
- **Efficiency:** Optimized for our specific use cases

**Alternatives Considered:**
- **GGUF:** Popular, but limited to uniform quantization
- **Safetensors:** Simple, but no quantization support
- **ONNX:** Standard, but complex and not optimized for LLMs
- **PyTorch:** Requires Python, multiple files

### Why AVX2 Instead of AVX-512?

**Decision:** Target AVX2 as the primary SIMD instruction set.

**Rationale:**
- **Wider support:** AVX2 is available on most modern CPUs
- **Good performance:** AVX2 provides significant speedups (4-8x for many operations)
- **Compatibility:** Works on both Intel and AMD CPUs
- **Future-proof:** AVX2 has been around since 2013

**Trade-offs:**
- AVX-512 provides better performance on supported CPUs
- Some operations could be faster with AVX-512
- May miss out on newer CPU features

**Future:** Add AVX-512 support as an optional optimization

### Why DirectX 12 for GPU?

**Decision:** Use DirectX 12 for GPU acceleration on Windows.

**Rationale:**
- **Windows focus:** DirectX is the native GPU API on Windows
- **Modern:** DirectX 12 provides better control and performance
- **Availability:** Available on all modern Windows versions
- **Explicit control:** Better for our use case than higher-level APIs

**Alternatives Considered:**
- **CUDA:** NVIDIA-only, but widely used in AI
- **Vulkan:** Cross-platform, but more complex (implemented as a dynamically-loaded backend in src/gpu_compute_vulkan.cpp)
- **Metal:** macOS-only
- **OpenCL:** Cross-platform, but less performant
- **SYCL:** Cross-platform, but newer and less mature

**Current:** DirectX 12 backend (Windows) + Vulkan backend (dynamically loaded) for cross-platform GPU acceleration

---

## 📊 Architecture Decisions

### Layered Architecture

**Decision:** Organize the codebase into clear layers.

**Rationale:**
- **Separation of concerns:** Each layer has a clear responsibility
- **Testability:** Layers can be tested independently
- **Extensibility:** New features can be added to appropriate layers
- **Maintainability:** Easier to understand and modify

**Layers:**
1. **Core:** Types, Tensor, Memory, Math
2. **Autograd:** Automatic differentiation
3. **Model:** Neural network components
4. **Quantization:** Format-specific kernels
5. **MoE:** Mixture of Experts
6. **GPU:** Hardware acceleration
7. **Training:** Trainer, Optimizer
8. **QUANT Format:** Binary format I/O
9. **Tools:** CLI utilities

### Engine Separation

**Decision:** Separate inference and training into different engines.

**Rationale:**
- **Different requirements:** Inference needs speed, training needs flexibility
- **Different optimizations:** Can optimize each for its specific use case
- **Clear separation:** Easier to maintain and extend
- **Optional features:** Users can choose which engine to use

**Trade-offs:**
- Some code duplication between engines
- More complex build system

### Format Planner

**Decision:** Use an importance-based format planner for quantization.

**Rationale:**
- **AWQ research:** Proven to work well in practice
- **Flexibility:** Can adapt to different target BPWs
- **Automation:** Removes manual tuning burden from users
- **Extensible:** Can add new allocation strategies

**How it works:**
1. Analyze model with calibration data
2. Score each weight block for importance
3. Allocate formats to hit target BPW
4. Optimize for quality within constraints

---

## 🔬 Technical Deep Dives

The following sections provide deep dives into specific technical areas:

### Memory Layout

See [Memory Management](memory_management.md) for details on:
- Memory allocation strategies
- Alignment requirements
- Contiguous vs. non-contiguous storage
- Custom allocators
- Arena allocation

### Kernel Optimization

See [Kernel Optimization](kernel_optimization.md) for details on:
- SIMD vectorization
- Cache optimization
- Loop unrolling
- Tiling strategies
- Numerical stability

### Autograd Implementation

See [Autograd Design](autograd_design.md) for details on:
- Computational graph construction
- Gradient computation
- Memory management for gradients
- Operation recording
- Performance considerations

### Quantization Strategy

See [Quantization Strategy](quantization_strategy.md) for details on:
- Mixed-format approach
- STE training
- Codebook learning
- Format selection
- Quality vs. size trade-offs

---

## 📝 How to Add a New Design Document

1. **Identify the need** - What decision or aspect needs documentation?
2. **Check for existing docs** - Is it already covered elsewhere?
3. **Choose a template** - Use one of the templates above
4. **Write the document** - Explain the what, why, and how
5. **Review** - Get feedback from other maintainers
6. **Commit** - Add to version control
7. **Update index** - Add link to this README

---

## 🎯 Best Practices for Design Documents

### 1. Be Clear

- Use simple, direct language
- Define terms and acronyms
- Provide examples
- Use diagrams when helpful

### 2. Be Comprehensive

- Cover all relevant aspects
- Include trade-offs and alternatives
- Document assumptions
- Note limitations

### 3. Be Current

- Update documents when designs change
- Mark outdated documents as deprecated
- Remove obsolete information

### 4. Be Accessible

- Keep documents in the repo (not external)
- Use markdown for formatting
- Keep file sizes reasonable
- Link to related documents

---

## 📚 Additional Resources

- **[Architecture Overview](../ARCHITECTURE.md)** - High-level system design
- **[Research Foundation](../RESEARCH.md)** - Research papers behind designs
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute
- **[API Reference](../API_REFERENCE.md)** - Complete API documentation
- **[Wiki Files](../wiki/files/_index.md)** - Per-file source documentation

---

*Last updated: July 12, 2026*
