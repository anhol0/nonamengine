#include "test_support.hpp"
#include "../src/model_io.hpp"

#include <limits>
#include <fstream>
#include <sstream>
#include <streambuf>

using Json = nlohmann::json;

namespace {

Json description() {
    return {
        {"format", "nonamengine"}, {"version", 1},
        {"weights", {{"file", "weights.bin"}, {"byte_order", "little"}, {"size", 12}}},
        {"layers", Json::array({{{"type", "dense"}, {"input_size", 2}, {"output_size", 1},
                                {"weights", "w"}, {"biases", "b"}}})},
        {"tensors", {
            {"w", {{"dtype", "float32"}, {"shape", {1, 2}}, {"offset", 0}, {"size", 8}}},
            {"b", {{"dtype", "float32"}, {"shape", {1, 1}}, {"offset", 8}, {"size", 4}}}
        }}
    };
}

std::string binary_data() {
    constexpr unsigned char bytes[] = {0, 0, 0, 64, 0, 0, 128, 191, 0, 0, 0, 63};
    return {reinterpret_cast<const char*>(bytes), sizeof(bytes)};
}

// Standard streambuf overrides model failures without filesystem races or OS devices.
class FaultyInput : public std::stringbuf {
public:
    enum class Failure { read, end_seek, tensor_seek, tell };
    explicit FaultyInput(Failure failure) : std::stringbuf(binary_data()), failure_(failure) {}
private:
    Failure failure_;
    std::streamsize xsgetn(char* destination, std::streamsize count) override {
        if(failure_ == Failure::read) return 0;
        return std::stringbuf::xsgetn(destination, count);
    }
    pos_type seekoff(off_type offset, std::ios::seekdir direction,
                     std::ios::openmode mode) override {
        if((failure_ == Failure::end_seek && direction == std::ios::end) ||
           (failure_ == Failure::tensor_seek && direction == std::ios::beg) ||
           (failure_ == Failure::tell && direction == std::ios::cur)) {
            return pos_type(off_type(-1));
        }
        return std::stringbuf::seekoff(offset, direction, mode);
    }
};

class FaultyOutput : public std::stringbuf {
public:
    explicit FaultyOutput(std::streamsize limit, bool fail_flush = false) :
        remaining_(limit), fail_flush_(fail_flush) {}
private:
    std::streamsize remaining_;
    bool fail_flush_;
    std::streamsize xsputn(const char* source, std::streamsize count) override {
        const auto written = std::stringbuf::xsputn(source, std::min(count, remaining_));
        remaining_ -= written;
        return written;
    }
    int sync() override {
        return fail_flush_ ? -1 : std::stringbuf::sync();
    }
};

class BrokenJsonInput : public std::streambuf {
    int_type underflow() override {
        throw std::runtime_error("Injected JSON read failure");
    }
};

class BadAtEof : public std::stringbuf {
public:
    explicit BadAtEof(const std::string& text) : std::stringbuf(text) {}
    std::istream* input = nullptr;
private:
    int_type underflow() override {
        input->setstate(std::ios::badbit);
        return traits_type::eof();
    }
};

void reading() {
    std::istringstream valid(binary_data());
    auto layers = model_io::read_layers(description(), valid);
    check(layers.size() == 1, "Stream load layer count");
    values(layers[0]->forward(tensor({2, 1}, {1, 3})), {-0.5f});
    for(auto failure : {FaultyInput::Failure::read, FaultyInput::Failure::end_seek,
                        FaultyInput::Failure::tensor_seek, FaultyInput::Failure::tell}) {
        FaultyInput buffer(failure);
        std::istream input(&buffer);
        throws<std::runtime_error>([&] { (void)model_io::read_layers(description(), input); });
    }
    std::istringstream truncated(binary_data().substr(0, 4));
    throws<std::invalid_argument>([&] { (void)model_io::read_layers(description(), truncated); });
    std::istringstream json_input(description().dump());
    check(model_io::read_description(json_input) == description(), "Stream JSON read");
    const auto json_text = description().dump();
    std::istringstream large_json(std::string(8192, ' ') + json_text);
    check(model_io::read_description(large_json) == description(), "Multi-buffer JSON read");
    std::istringstream boundary_json(json_text + std::string(4096 - json_text.size(), ' '));
    check(model_io::read_description(boundary_json) == description(), "Exact-buffer JSON read");
    BrokenJsonInput buffer;
    std::istream broken(&buffer);
    throws<std::runtime_error>([&] { (void)model_io::read_description(broken); });
    std::istringstream bad_json(description().dump());
    bad_json.setstate(std::ios::badbit);
    throws<std::runtime_error>([&] { (void)model_io::read_description(bad_json); });
    BadAtEof bad_at_eof(description().dump());
    std::istream end_failure(&bad_at_eof);
    bad_at_eof.input = &end_failure;
    throws<std::runtime_error>([&] { (void)model_io::read_description(end_failure); });
}

void writing() {
    std::vector<std::unique_ptr<Layer>> layers;
    layers.push_back(std::make_unique<Dense>(tensor({1, 2}, {2, -1}), tensor({1, 1}, {0.5f})));
    std::ostringstream valid;
    auto metadata = model_io::write_weights(layers, valid, "weights.bin", 1);
    check(valid.str() == binary_data(), "Exact stream binary data");
    check(metadata["weights"]["size"] == 12, "Stream binary size");
    for(std::streamsize limit : {0, 5}) {
        FaultyOutput buffer(limit);
        std::ostream output(&buffer);
        throws<std::runtime_error>([&] {
            (void)model_io::write_weights(layers, output, "weights.bin", 1);
        });
    }
    FaultyOutput buffer(1000, true);
    std::ostream output(&buffer);
    throws<std::runtime_error>([&] {
        (void)model_io::write_weights(layers, output, "weights.bin", 1);
    });
    check(buffer.str() == binary_data(), "Flush failure must occur after writes");
    std::ostringstream json_output;
    model_io::write_description(json_output, metadata);
    check(Json::parse(json_output.str()) == metadata, "Stream JSON export");
    FaultyOutput short_json(5);
    std::ostream short_output(&short_json);
    throws<std::runtime_error>([&] { model_io::write_description(short_output, metadata); });
    FaultyOutput flush_json(10000, true);
    std::ostream flush_output(&flush_json);
    throws<std::runtime_error>([&] { model_io::write_description(flush_output, metadata); });
}

void offsets() {
    constexpr auto maximum = std::numeric_limits<uint64_t>::max();
    check(model_io::tensor_end_offset(0, 0) == 0, "Empty interval");
    check(model_io::tensor_end_offset(2, 4) == 12, "Normal interval");
    check(model_io::tensor_end_offset(maximum / 4, 0) == maximum - 3, "Largest byte count");
    check(model_io::tensor_end_offset(0, maximum) == maximum, "Zero-length end interval");
    throws<std::overflow_error>([&] { (void)model_io::tensor_end_offset(maximum / 4 + 1, 0); });
    throws<std::overflow_error>([&] { (void)model_io::tensor_end_offset(1, maximum - 3); });
}

void closing() {
    TestDirectory directory;
    std::ofstream output(directory.path / "closed.bin", std::ios::binary);
    check(bool(output), "Could not open close-test fixture");
    model_io::close_output(output);
    check(!output.is_open(), "Output file was not closed");
    // Closing a closed file is a deterministic failure of the real filebuf.
    throws<std::runtime_error>([&] { model_io::close_output(output); });
    std::ofstream failed(directory.path / "failed.bin", std::ios::binary);
    check(bool(failed), "Could not open failure fixture");
    failed.setstate(std::ios::badbit);
    throws<std::runtime_error>([&] { model_io::close_output(failed); });
    check(!failed.is_open(), "Failed output still needs to release its file");
}

} // namespace

int main(int argc, char** argv) {
    return run_test([&] {
        check(argc == 2, "Expected case name");
        const std::string name = argv[1];
        if(name == "reading") reading();
        else if(name == "writing") writing();
        else if(name == "offsets") offsets();
        else if(name == "closing") closing();
        else throw std::runtime_error("Unknown case");
    });
}
