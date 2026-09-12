#include "quant/random.h"
#include <cmath>
#include <limits>
#include <random>
#include <cstdlib>

namespace quant {

// ── GlobalSeedManager ──
std::atomic<uint64_t> GlobalSeedManager::base_seed_{0};
std::atomic<uint64_t> GlobalSeedManager::counter_{0};

uint64_t GlobalSeedManager::get_seed() {
    uint64_t seed = base_seed_.load(std::memory_order_acquire);
    if (seed != 0) return seed;
    const char* env = std::getenv("QUANT_SEED");
    if (env) {
        uint64_t v = (uint64_t)std::atoll(env);
        if (v == 0) v = 1; // 0 means "use entropy"
        uint64_t expect = 0;
        base_seed_.compare_exchange_strong(expect, v);
        return base_seed_.load(std::memory_order_acquire);
    }
    {
        std::random_device rd;
        uint64_t v = ((uint64_t)rd() << 32) | rd();
        if (v == 0) v = 1;
        uint64_t expect = 0;
        base_seed_.compare_exchange_strong(expect, v);
    }
    return base_seed_.load(std::memory_order_acquire);
}

uint64_t GlobalSeedManager::next_seed() {
    // Counter is atomic: every thread gets a unique stream offset.
    uint64_t n = counter_.fetch_add(1, std::memory_order_relaxed) + 1;
    return get_seed() + n;
}

bool GlobalSeedManager::is_deterministic() {
    const char* env = std::getenv("QUANT_SEED");
    return env != nullptr && std::atoll(env) != 0;
}

static inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

RNG::RNG(uint64_t seed_value) { seed(seed_value); }

void RNG::seed(uint64_t seed_value) {
    uint64_t tmp = seed_value;
    s[0] = splitmix64(tmp);
    s[1] = splitmix64(tmp);
}

uint64_t RNG::next_u64() {
    uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s[1] = rotl(s1, 37);
    return result;
}

uint32_t RNG::next_u32() { return (uint32_t)next_u64(); }

float RNG::uniform() {
    return (next_u64() >> 11) * (1.0f / 9007199254740992.0f);
}

float RNG::normal() {
    float u1 = uniform();
    float u2 = uniform();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.283185307179586f * u2);
}

int RNG::uniform_int(int lo, int hi) {
    uint64_t range = (uint64_t)(hi - lo + 1);
    uint64_t limit = UINT64_MAX - (UINT64_MAX % range);
    uint64_t val;
    do { val = next_u64(); } while (val >= limit);
    return lo + (int)(val % range);
}

// ── J1 seed registry (D7 W4) — append-only implementations ──
uint64_t make_seed(uint64_t base_seed, uint64_t stream_id, uint64_t counter) noexcept {
    uint32_t words[6];
    words[0] = static_cast<uint32_t>(base_seed & 0xffffffffu);
    words[1] = static_cast<uint32_t>(base_seed >> 32);
    words[2] = static_cast<uint32_t>(stream_id & 0xffffffffu);
    words[3] = static_cast<uint32_t>(stream_id >> 32);
    words[4] = static_cast<uint32_t>(counter & 0xffffffffu);
    words[5] = static_cast<uint32_t>(counter >> 32);
    std::seed_seq seq(words, words + 6);
    uint32_t out[2] = {0, 0};
    seq.generate(out, out + 2);
    uint64_t mixed = (static_cast<uint64_t>(out[0]) << 32) | out[1];
    return mixed != 0 ? mixed : 1u;
}

RNG make_rng(uint64_t base_seed, uint64_t stream_id, uint64_t counter) {
    return RNG(make_seed(base_seed, stream_id, counter));
}

uint64_t resolve_base_seed(uint64_t cli_seed) noexcept {
    if (cli_seed != 0) return cli_seed;
    const char* env = std::getenv("QUANT_SEED");
    if (env) {
        long long v = std::atoll(env);
        if (v != 0) return static_cast<uint64_t>(v);
    }
    return 42;
}

} // namespace quant
