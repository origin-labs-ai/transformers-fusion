#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

namespace quant {

// ── GlobalSeedManager — entropy-based seed source ──
// Uses std::random_device by default. Override with QUANT_SEED env var.
// QUANT_SEED=0 disables entropy for full deterministic reproducibility.
// BUGFIX (bug census): base_seed_/counter_ were plain statics read/written
// without synchronization (race → duplicate seeds across threads). Atomics
// now; init via compare_exchange so exactly one thread seeds entropy.
class GlobalSeedManager {
public:
    static uint64_t get_seed();
    static uint64_t next_seed();
    static bool is_deterministic();

private:
    static std::atomic<uint64_t> base_seed_;
    static std::atomic<uint64_t> counter_;
};

class RNG {
public:
    explicit RNG(uint64_t seed = GlobalSeedManager::get_seed());

    void seed(uint64_t s);

    float uniform();
    float normal();
    int uniform_int(int lo, int hi);

    uint64_t next_u64();
    uint32_t next_u32();

private:
    uint64_t s[2];

    static uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

// ── J1 seed registry (D7 W4) — central deterministic seed plumbing ──
// Base (Config.seed / --seed / QUANT_SEED) + stream_id + counter
// (epoch/step/block) mixed via std::seed_seq (see src/core/random.cpp).
// Mirrors transformer.cpp `42 + block_idx*7919` and trainer_data.cpp
// `42+epoch` seed+offset pattern, centralized so streams never collide.
// Append-only: existing GlobalSeedManager/RNG signatures unchanged.
enum class SeedStream : uint64_t {
    InitBlock           = 1,  // transformer init_uniform reference (no code change)
    ShuffleEpoch        = 2,  // trainer_data shuffle reference (no code change)
    GeneratorSampler    = 11, // src/inference/generator.cpp Generator ctor
    SpeculativeVerify   = 12, // SpeculativeDecoder verify/rng_ stream
    SpeculativeFallback = 13, // SpeculativeDecoder fallback-logits stream
    EngineSampler       = 14, // engines/inference/inference.cpp sampler_
    ContinualReplay     = 20, // continual ExperienceReplay sample_batch
    ContinualEntropy    = 21, // continual apply_entropy_floor
    DdpNoise            = 22  // ddp inject_gradient_noise (counter = step)
};

uint64_t make_seed(uint64_t base_seed, uint64_t stream_id, uint64_t counter) noexcept;
inline uint64_t make_seed(uint64_t base_seed, SeedStream stream, uint64_t counter) noexcept {
    return make_seed(base_seed, static_cast<uint64_t>(stream), counter);
}
RNG make_rng(uint64_t base_seed, uint64_t stream_id, uint64_t counter);
inline RNG make_rng(uint64_t base_seed, SeedStream stream, uint64_t counter) {
    return make_rng(base_seed, static_cast<uint64_t>(stream), counter);
}
// Effective base: CLI --seed (non-zero wins) > QUANT_SEED env (non-zero) > 42 (Config::seed default).
// Keeps deterministic tests pinned when no override is given.
uint64_t resolve_base_seed(uint64_t cli_seed) noexcept;

} // namespace quant
