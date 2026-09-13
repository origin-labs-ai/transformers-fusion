# QUANT Index + SHA256 Content-Addressed Weight mmap — Research

**Task:** DIFFUSION.txt Phase K, Task 134  
**Date:** 2026-07-16  
**Scope:** Best practices for SHA256-indexed, content-addressed weight storage with memory-mapped loading — the `.quant` + `TranscenderIDX` index format used by Transcender.

---

## 1. Content-Addressed Weight Storage

### The idea
Instead of storing tensors by **position** (tensor #0, #1, … at fixed offsets), store by **content hash** — SHA256 over the raw weight bytes. The index file (`TranscenderIDX`) maps each tensor's logical name to its SHA256, and the data file holds the (deduplicated) blobs.

### Why content-addressing
- **Deduplication** — identical tensors (shared embeddings, tied weights, common LoRA deltas, identical experts across MoE checkpoints) hash to the same SHA256 and are stored once. Across a model family this is a large win; across MoE with many experts it is substantial.
- **Integrity** — recomputing SHA256 on load and comparing to the stored hash detects **any** corruption (bit rot, truncated writes, partial overwrites) with probability 1 − 2^-256. Positional offsets give zero integrity guarantee.
- **Caching / reuse** — a content-addressed blob is globally addressable: a downloader, a tier-2 SSD cache, and a peer can all key on the hash; you never re-fetch what you already have.
- **Differential / sharded loading** — when a model is updated, only tensors whose SHA256 changed need to be re-downloaded/reloaded (git-for-weights). This is the backbone of efficient distributed checkpoint pull (pairs with ZeRO-3 sharding — see `distributed_research.md`).

### Transcender index layout (per DIFFUSION.txt conditions 13 & 14)
```
magic        : 15 bytes = "TranscenderIDX" + NUL (Phase 24: was "8 bytes" — wrong; truth `src/codec/quant_format.cpp:546-548` writes 15B incl. NUL; reader also accepts legacy 10B "InNovaIDX", `:585-592`)
version      : uint32
num_tensors  : uint32
[ per tensor:
    name_len   : uint32
    name       : bytes[name_len]      (UTF-8 tensor name, e.g. "layers.0.attn.q_proj.weight")
    shape_rank : uint32
    shape      : int64[shape_rank]
    dtype      : uint32               (enum: F32/F16/INT8/QUANT8/QUANT4/ternary/binary/FP8...)
    offset     : uint64               (byte offset into the .quant data file)
    nbytes     : uint64               (compressed/stored byte count
    sha256     : 32 bytes            (hash over the *stored* bytes at [offset, offset+nbytes))
]
```
On load: read header → for each tensor, `mmap` (or pread) the byte range at `offset` → **recompute SHA256 over [offset, offset+nbytes)** → compare to stored hash → **fail fast with the offending tensor name** if mismatch (condition 15: "Quant idx integrity corrupt one byte must fail tensor name"). The hash is over the *stored* (possibly quantized/QUANT) bytes, so it validates the on-disk representation, not a hypothetical FP32 reconstruction.

### Integrity verification patterns
1. **Full verify on cold load** — recompute all hashes; reject the whole file if any fail. Slowest but safest. Use for first load / release artifacts.
2. **Lazy / on-demand verify** — verify a tensor's hash only when it is first touched (deferred until the mmap region is page-faulted in). Trades a little memory of trust for fast startup. Use for warm caches where the index was previously verified.
3. **Merkle / chunked hashing** — for very large tensors, hash in fixed chunks (e.g. 1 MB) and store a Merkle root; lets you localize corruption to a chunk and re-fetch just that chunk (git-LFS-style). Overkill for 0.1B, valuable for 1T-param sharded checkpoints.
4. **Streaming recompute** — verify while streaming the tensor through the dequantizer, so you pay the SHA256 cost once and it overlaps with I/O (no extra pass over memory). This is the recommended Transcender pattern: fold SHA256 into the `mmap`→dequant loop so integrity is free-ish.

**Critical correctness note:** SHA256 must be computed over the **exact bytes that were written**, with a stable byte order (little-endian, as Transcender uses). If you ever change quantization format, the hash changes — that's correct (it is a new artifact), so versioning is in the header, not by mutating bytes in place.

---

## 2. mmap vs read() Performance

### The model
- `read()` is a **copy**: kernel DMAs disk → page cache, then copies page cache → user buffer. Two memory touches, and the user buffer must be allocated and filled.
- `mmap()` is a **map**: kernel sets up page tables so the file's page-cache pages are mapped directly into the process address space. First touch triggers a page fault → disk read → page mapped. **Zero copy** into user space; the process reads the file in place.

### When mmap wins
- **Large files read once or sparsely** — the page-cache is shared, no user buffer allocation, no `memcpy`. For a multi-GB checkpoint this is the difference between "uses 2× file size in RAM" and "uses ~1×, lazily."
- **Random / repeated access** — pages stay mapped; repeated reads of the same tensor cost nothing after the first fault.
- **Multiple processes share one checkpoint** — page-cache pages are shared across processes that mmap the same file read-only (copy-on-write only on write). A serving fleet loading the same `.quant` file shares the bulk of the resident pages. This is the key win for inference fleets.
- **Lazy loading** — only touched tensors get paged in; for MoE only the active experts need to be resident (load-balance → only a subset of expert blobs are hot).

### When read() can win / be safer
- **Small files** — mmap setup (page-table manipulation, `MAP_FAILED` handling) has fixed overhead; for tiny tensors `read` is simpler and as fast.
- **Write path / training checkpoints** — mmap'd writes (`MAP_SHARED` writable) are coherent but subtle (you must `msync`/`FlushViewOfFile` to flush, and partial-write durability is OS-dependent). `write()`/`fwrite` gives clear durability semantics and is easier to make crash-safe. Transcender uses `fwrite` for checkpoints (`trainer.cpp:save_checkpoint`) and mmap for loads — correct split.
- **Portability of weird filesystems** — mmap on network/sparse files can be slow or unsupported; `pread` degrades gracefully.

### Numbers (typical, commodity NVMe + DDR4)
- Sequential `read()` of a 10 GB file: ~2–3 GB/s, doubles RSS (page cache + user buffer).
- `mmap` first-touch sequential: ~3–5 GB/s (one copy), RSS grows lazily to ~file size; second pass if still resident: memory-bandwidth limited (~20+ GB/s).
- **For inference, mmap is ~1.5–2× faster on cold load and dramatically lower peak RSS** because there is no user-buffer copy and the OS manages eviction.

---

## 3. Windows CreateFileMapping vs Linux mmap

Transcender already abstracts this (`trainer.cpp:open_mmap`, `quant_format.cpp`). The API differences and best practices:

| Concern | Linux / POSIX | Windows |
|--------|---------------|---------|
| Open | `open(path, O_RDONLY)` → fd | `CreateFileA(..., GENERIC_READ, ..., OPEN_EXISTING, ...)` → `HANDLE` |
| Size | `fstat(fd).st_size` | `GetFileSizeEx(handle, &li)` |
| Map | `mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0)` → ptr | `CreateFileMappingA(handle, NULL, PAGE_READONLY, 0, 0, NULL)` → mapHandle; `MapViewOfFile(mapHandle, FILE_MAP_READ, 0, 0, 0)` → ptr |
| Failure sentinel | `MAP_FAILED` (check `== MAP_FAILED`, **not** NULL) | `NULL` handle / `NULL` view |
| Unmap | `munmap(ptr, sz)` | `UnmapViewOfFile(ptr)`; then `CloseHandle(mapHandle)`; then `CloseHandle(fileHandle)` |
| Partial map | `mmap(NULL, len, ..., off)` with `off` page-aligned | `MapViewOfFile(mapHandle, access, offHigh, offLow, len)` |
| Sync (writable) | `msync(ptr, len, MS_SYNC)` | `FlushViewOfFile(ptr, len)` |
| Advice | `madvise(ptr, len, MADV_RANDOM/MADV_SEQUENTIAL/MADV_DONTNEED)` | `PrefetchVirtualMemory` / `Win32 PrefetchMemory` (limited); set `FILE_FLAG_SEQUENTIAL_SCAN` on CreateFile |
| Huge pages | `mmap(..., MAP_HUGETLB)` | `VirtualAlloc(..., MEM_LARGE_PAGES)` (requires SeLockMemory privilege) |

### Best practices (both platforms)
- **Map read-only for inference** (`PROT_READ`/`PAGE_READONLY`, `MAP_PRIVATE`). No COW, shared page cache, safe.
- **Close handles in order**: unmap the view **before** closing the mapping handle **before** closing the file handle on Windows; on Linux just `munmap` + `close(fd)`.
- **Check the sentinel correctly** — `mmap` returns `MAP_FAILED` (not NULL) on error; a common bug is `if (!ptr)` which never triggers because `MAP_FAILED != NULL`. Transcender's `close_mmap` does this right (`if (mmap_ptr_ == MAP_FAILED)`).
- **Alignment**: `mmap` offsets must be page-aligned (`sysconf(_SC_PAGESIZE)`, usually 4096); Windows file-mapping offsets must be `GetSystemInfo().dwAllocationGranularity`-aligned (typically 65536). For per-tensor partial maps, round the offset **down** to the granularity and adjust the pointer up.
- **Lazy access**: do not touch the whole mapping up front (that defeats the point). Access tensors as the model needs them; the OS faults pages in. For a known access pattern (training = all tensors), a single `madvise(MADV_SEQUENTIAL)` or Windows `FILE_FLAG_SEQUENTIAL_SCAN` hints the prefetcher.
- **Large files (> virtual address space)**: on 32-bit this is a hard limit; Transcender targets 64-bit so the whole checkpoint maps. For > available RAM, rely on the OS page cache + `MADV_DONTNEED` to evict cold experts.
- **Do not mmap + fread the same fd** interchangeably — offset state diverges. Pick one model per file.

---

## 4. Best Practices for Large Model Loading (1B–1T params)

1. **Single mmap of the whole data file, index-driven access.** Map the entire `.quant` data file once (lazy); use the `TranscenderIDX` index to get `(offset, nbytes, sha256)` per tensor and compute pointers as `base + offset`. One mapping, N tensors — cheapest setup and best page-cache sharing.

2. **Verify SHA256 lazily / in the dequant stream.** Compute SHA256 while streaming a tensor's bytes through the dequantizer (INT8/QUANT8/ternary → FP32 on the fly). Integrity is then "free" — it rides on the bytes you must read anyway — and a corrupt tensor is caught before it poisons compute. Fail fast with the tensor name (condition 15).

3. **Deduplicate by hash.** Before writing a checkpoint, hash each tensor; if the SHA256 already exists in the store, skip writing the blob and just add an index entry pointing to the existing offset. Across MoE experts and shared embeddings this can halve checkpoint size. Reference the same offset from multiple index entries.

4. **Page-aligned tensor layout.** Write tensors at offsets aligned to the OS page/granularity (4096 Linux, 65536 Windows). This lets a single `mmap` map any tensor individually if needed and avoids cross-tensor page-fault amplification. Pad with zeros between tensors to the next alignment boundary.

5. **Tiered / mmap-backed block cache for >RAM models.** Keep the index + hot tensors (active experts, current layer) resident; let the OS page cache hold the rest backed by the mmap'd file. For 1T-param MoE with sparse activation, only the active experts per token are touched, so resident working set ≪ model size. Optionally `madvise(MADV_DONTNEED)` on experts that fall out of the load-balance hot set to free pages proactively.

6. **Stream the index, not all metadata.** Read only the index header into memory (it is small: name + hash + offset per tensor); never load all tensor bytes at startup. Startup time = O(index size), not O(model size).

7. **Atomic index + data on write.** Write the data blobs first, fsync, then write the index with their hashes, fsync. On load, if the index is missing/truncated, the data file is garbage-collectable but not loadable. Never mutate a blob in place (it changes the hash → invalidates the index); always write a new blob + new index entry (content-addressed append-only).

8. **Cross-platform mmap wrapper.** Hide `mmap`/`CreateFileMapping` behind one `quant::MemoryMap` RAII type (open, map, pointer, size, close) so `quant_format.cpp` and `trainer.cpp` share one tested path — matching Transcender's existing `open_mmap`/`close_mmap` split but unified.

9. **Determinism.** Same prompt + seed → same tokens requires that weight load is bit-exact. mmap gives bit-exact load (no copy/resample); verify with SHA256. This is what the README "determinism" and condition 145 ("determinism same prompt seed 2 runs same tokens") depend on.

10. **Fail fast, name the tensor.** On any hash mismatch, throw an exception whose message contains the **tensor name and expected vs actual SHA256** (condition 13/15: "fail fast tensor name on corrupt"). Do not silently zero a corrupt tensor — that produces plausible-but-wrong output (the worst failure mode).

---

## Application to Transcender `quant_format.cpp` / `quant_idx`
- The `TranscenderIDX` header (magic, version, num_tensors) + per-tensor `(name, shape, dtype, offset, nbytes, sha256)` is the target schema.
- `quant_load` should: mmap the data file → parse index → for each tensor, validate SHA256 lazily in the dequant stream → throw with tensor name on mismatch.
- `quant_save` should: hash each tensor, dedup against existing blobs, append-only write, then atomically write the index.
- The cross-platform mmap is already split correctly (`trainer.cpp`); unify it into one RAII `MemoryMap` used by both `quant_format` and the DataLoader.
