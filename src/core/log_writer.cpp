#include "quant/log_writer.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace quant {

// C21: TensorBoard-compatible event writer
// Writes TF Events file format: [uint64 length][uint32 crc][Event proto][uint32 crc]
// Uses a minimal binary format compatible with TensorBoard's event reading.

static int64_t current_wall_time() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static uint32_t masked_crc32c(const uint8_t* data, size_t len) {
    static uint32_t table[256];
    static std::once_flag table_flag;
    std::call_once(table_flag, []() {
        const uint32_t poly = 0x82F63B78;
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t crc = i;
            for (int j = 0; j < 8; j++)
                crc = (crc >> 1) ^ (poly & (-(int32_t)(crc & 1)));
            table[i] = crc;
        }
    });
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++)
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xFF];
    crc ^= 0xFFFFFFFF;
    return ((crc >> 15) | (crc << 17)) + 0xa282ead8UL;
}

static uint32_t masked_crc32c(const std::string& s) {
    return masked_crc32c((const uint8_t*)s.data(), s.size());
}

EventWriter::EventWriter(const std::string& logdir) : logdir_(logdir) {
    std::filesystem::create_directories(logdir_);
    // BUGFIX (bug census): std::localtime thread-unsafe; result was unused
    // anyway (filename uses current_wall_time()). Dropped the dead call.
    std::ostringstream oss;
    oss << logdir_ << "/events.out.tfevents." << current_wall_time()
        << ".localhost";
    file_.open(oss.str(), std::ios::binary);
}

EventWriter::~EventWriter() {
    flush();
    if (file_.is_open()) file_.close();
}

// TensorBoard protobuf Event writer
// Implements minimal protobuf encoding for tf.Event proto:
//   message Event { double wall_time=1; int64 step=2; Summary summary=5; }
//   message Summary { repeated SummaryValue value=1; }
//   message SummaryValue { string tag=1; float simple_value=2; }

static void encode_varint(uint64_t val, std::string& out) {
    while (val > 0x7F) {
        out.push_back(static_cast<char>((val & 0x7F) | 0x80));
        val >>= 7;
    }
    out.push_back(static_cast<char>(val));
}

static void encode_field(int field, int wire_type, std::string& out) {
    encode_varint(static_cast<uint64_t>((field << 3) | wire_type), out);
}

static void encode_double_field(int field, double val, std::string& out) {
    encode_field(field, 1, out);
    const auto* p = reinterpret_cast<const uint8_t*>(&val);
    out.append(reinterpret_cast<const char*>(p), 8);
}

static void encode_int64_field(int field, int64_t val, std::string& out) {
    encode_field(field, 0, out);
    encode_varint(static_cast<uint64_t>(val), out);
}

static void encode_float_field(int field, float val, std::string& out) {
    encode_field(field, 5, out);
    const auto* p = reinterpret_cast<const uint8_t*>(&val);
    out.append(reinterpret_cast<const char*>(p), 4);
}

static void encode_bytes_field(int field, const std::string& data, std::string& out) {
    encode_field(field, 2, out);
    encode_varint(data.size(), out);
    out.append(data);
}

static void encode_submsg(int field, const std::string& msg, std::string& out) {
    encode_field(field, 2, out);
    encode_varint(msg.size(), out);
    out.append(msg);
}

void EventWriter::write_event(const std::string& tag, float value, int step, int64_t wall_time) {
    if (!file_.is_open()) return;
    std::lock_guard<std::mutex> lock(mtx_);

    // SummaryValue: tag=1 (string), simple_value=2 (float)
    std::string sv;
    encode_bytes_field(1, tag, sv);
    encode_float_field(2, value, sv);

    // Summary: value=1 (repeated SummaryValue)
    std::string summary;
    encode_submsg(1, sv, summary);

    // Event: wall_time=1 (double), step=2 (int64), summary=5 (Summary)
    std::string event;
    encode_double_field(1, static_cast<double>(wall_time), event);
    encode_int64_field(2, step, event);
    encode_submsg(5, summary, event);

    // Write TF Events record: [uint64 length][uint32 crc][Event bytes][uint32 crc]
    uint64_t len = static_cast<uint64_t>(event.size());
    uint32_t crc1 = masked_crc32c(reinterpret_cast<const uint8_t*>(&len), sizeof(len));
    uint32_t crc2 = masked_crc32c(event);
    file_.write(reinterpret_cast<const char*>(&len), sizeof(len));
    file_.write(reinterpret_cast<const char*>(&crc1), sizeof(crc1));
    file_.write(event.data(), static_cast<std::streamsize>(event.size()));
    file_.write(reinterpret_cast<const char*>(&crc2), sizeof(crc2));
}

void EventWriter::write_scalar(const std::string& tag, float value, int step) {
    write_event(tag, value, step, current_wall_time());
}

void EventWriter::write_scalars(const TrainMetrics& metrics, int step) {
    write_scalar("loss", metrics.loss, step);
    write_scalar("perplexity", metrics.perplexity, step);
    write_scalar("grad_norm", metrics.grad_norm, step);
    write_scalar("learning_rate", metrics.learning_rate, step);
    write_scalar("tokens_per_sec", (float)metrics.tokens_per_sec, step);
    if (metrics.val_loss > 0) {
        write_scalar("val_loss", metrics.val_loss, step);
        write_scalar("val_perplexity", metrics.val_perplexity, step);
    }
    flush();
}

void EventWriter::flush() {
    if (file_.is_open()) file_.flush();
}

// C22: WandB-compatible lightweight logger
WandBLogger::WandBLogger(const std::string& project, const std::string& run_name) {
    run_name_ = run_name.empty() ? "run_" + std::to_string(current_wall_time()) : run_name;
    std::string dir = "wandb/" + project + "/" + run_name_;
    std::filesystem::create_directories(dir);
    file_.open(dir + "/metrics.json", std::ios::app);
}

WandBLogger::~WandBLogger() { flush(); if (file_.is_open()) file_.close(); }

void WandBLogger::log(const std::string& key, float value, int step) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!file_.is_open()) return;
    file_ << "{\"step\": " << step << ", \"" << key << "\": " << value << "}\n";
}

void WandBLogger::log_metrics(const TrainMetrics& metrics, int step) {
    log("loss", metrics.loss, step);
    log("perplexity", metrics.perplexity, step);
    log("grad_norm", metrics.grad_norm, step);
    log("learning_rate", metrics.learning_rate, step);
    log("tokens_per_sec", (float)metrics.tokens_per_sec, step);
    if (metrics.val_loss > 0) log("val_loss", metrics.val_loss, step);
    if (metrics.val_perplexity > 0) log("val_perplexity", metrics.val_perplexity, step);
    flush();
}

void WandBLogger::flush() { if (file_.is_open()) file_.flush(); }

} // namespace quant
