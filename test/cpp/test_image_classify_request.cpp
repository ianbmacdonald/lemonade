// Request parsing and limits for POST /v1/images/classify
// (lemon::image_classify). Pure functions, no server.

#include "lemon/image_classify_request.h"

#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace lemon::image_classify;

namespace {

int failures = 0;

void check(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

const std::string kJpeg = std::string("\xFF\xD8\xFF\xE0", 4) + "rest-of-jpeg";
const std::string kPng = std::string("\x89PNG\r\n\x1A\n", 8) + "rest-of-png";

std::string b64(const std::string& in) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    std::size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        unsigned v = (static_cast<unsigned char>(in[i]) << 16) |
                     (static_cast<unsigned char>(in[i + 1]) << 8) |
                     static_cast<unsigned char>(in[i + 2]);
        out += t[(v >> 18) & 63];
        out += t[(v >> 12) & 63];
        out += t[(v >> 6) & 63];
        out += t[v & 63];
    }
    if (i + 1 == in.size()) {
        unsigned v = static_cast<unsigned char>(in[i]) << 16;
        out += t[(v >> 18) & 63];
        out += t[(v >> 12) & 63];
        out += "==";
    } else if (i + 2 == in.size()) {
        unsigned v = (static_cast<unsigned char>(in[i]) << 16) |
                     (static_cast<unsigned char>(in[i + 1]) << 8);
        out += t[(v >> 18) & 63];
        out += t[(v >> 12) & 63];
        out += t[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

std::string json_body(const std::string& image, const std::string& extra = "") {
    return "{\"model\":\"m\",\"image\":\"" + image + "\"" + extra + "}";
}

void test_base64() {
    std::string out;
    check("base64: round-trips 1/2/3-byte tails",
          strict_base64_decode(b64("a"), out) && out == "a" &&
              strict_base64_decode(b64("ab"), out) && out == "ab" &&
              strict_base64_decode(b64("abc"), out) && out == "abc");
    check("base64: binary round-trips", strict_base64_decode(b64(kJpeg), out) && out == kJpeg);
    check("base64: whitespace ignored",
          strict_base64_decode("YW\r\nJj\t", out) && out == "abc");
    check("base64: bad alphabet rejected", !strict_base64_decode("YW*j", out));
    check("base64: url-safe alphabet rejected", !strict_base64_decode("YW-_", out));
    check("base64: length not multiple of 4 rejected", !strict_base64_decode("YWJ", out));
    check("base64: interior padding rejected", !strict_base64_decode("YW=j", out));
    check("base64: '=' then data rejected", !strict_base64_decode("Y=Jj", out));
    check("base64: triple padding rejected", !strict_base64_decode("Y===", out));
    check("base64: empty rejected", !strict_base64_decode("", out));
}

void test_sniff() {
    check("sniff: JPEG", sniff_image_mime(kJpeg) == "image/jpeg");
    check("sniff: PNG", sniff_image_mime(kPng) == "image/png");
    check("sniff: GIF rejected", sniff_image_mime("GIF89a....").empty());
    check("sniff: empty rejected", sniff_image_mime("").empty());
    check("sniff: truncated PNG signature rejected",
          sniff_image_mime(std::string("\x89PNG", 4)).empty());
}

void test_json() {
    auto ok = parse_json(json_body(b64(kJpeg), ",\"top_k\":3"));
    check("json: base64 JPEG accepted", ok.ok() && ok.bytes == kJpeg && ok.mime == "image/jpeg");
    check("json: params carry model and top_k, never the image",
          ok.params.value("model", "") == "m" && ok.params.value("top_k", 0) == 3 &&
              !ok.params.contains("image"));

    auto data_url = parse_json(json_body("data:image/png;base64," + b64(kPng)));
    check("json: PNG data URL accepted", data_url.ok() && data_url.mime == "image/png");
    auto data_upper = parse_json(json_body("DATA:IMAGE/JPEG;BASE64," + b64(kJpeg)));
    check("json: data URL scheme is case-insensitive", data_upper.ok());
    check("json: non-image data URL rejected",
          parse_json(json_body("data:text/plain;base64," + b64("hello"))).status == 400);
    check("json: non-base64 data URL rejected",
          parse_json(json_body("data:image/png," + b64(kPng))).status == 400);
    check("json: http URL rejected",
          parse_json(json_body("http://example.com/a.jpg")).status == 400);
    auto https = parse_json(json_body("https://example.com/a.jpg"));
    check("json: https URL rejected with a message naming the alternative",
          https.status == 400 && https.error.find("data: URL") != std::string::npos);
    check("json: bad base64 rejected", parse_json(json_body("@@@@")).status == 400);
    check("json: GIF bytes rejected",
          parse_json(json_body(b64("GIF89a-not-supported"))).status == 400);
    check("json: missing image rejected", parse_json("{\"model\":\"m\"}").status == 400);
    check("json: non-string image rejected",
          parse_json("{\"model\":\"m\",\"image\":5}").status == 400);
    check("json: non-string model rejected",
          parse_json("{\"model\":1,\"image\":\"" + b64(kJpeg) + "\"}").status == 400);
    check("json: malformed JSON rejected", parse_json("{").status == 400);
    check("json: non-object rejected", parse_json("[]").status == 400);

    const std::string img = b64(kJpeg);
    check("top_k: 0 rejected", parse_json(json_body(img, ",\"top_k\":0")).status == 400);
    check("top_k: 1 accepted", parse_json(json_body(img, ",\"top_k\":1")).ok());
    check("top_k: 1000000 accepted", parse_json(json_body(img, ",\"top_k\":1000000")).ok());
    check("top_k: 1000001 rejected",
          parse_json(json_body(img, ",\"top_k\":1000001")).status == 400);
    check("top_k: float rejected", parse_json(json_body(img, ",\"top_k\":2.5")).status == 400);
    check("top_k: string rejected", parse_json(json_body(img, ",\"top_k\":\"3\"")).status == 400);
    check("top_k: negative rejected", parse_json(json_body(img, ",\"top_k\":-1")).status == 400);

    std::string at_limit = kJpeg;
    at_limit.resize(kMaxImageBytes, 'x');
    check("size: exactly 16 MiB accepted", parse_json(json_body(b64(at_limit))).ok());
    std::string over = at_limit + "x";
    auto big = parse_json(json_body(b64(over)));
    check("size: 16 MiB + 1 is 413", big.status == 413);
    check("size: empty image is 400", parse_json(json_body("")).status == 400);
}

void test_form() {
    const std::vector<std::string_view> one = {kJpeg};
    auto ok = from_form(std::string("m"), std::string("3"), one);
    check("form: one part accepted", ok.ok() && ok.bytes == kJpeg &&
                                         ok.params.value("top_k", 0) == 3 &&
                                         ok.params.value("model", "") == "m");
    auto implicit = from_form(std::nullopt, std::nullopt, one);
    check("form: model and top_k optional",
          implicit.ok() && !implicit.params.contains("model") && !implicit.params.contains("top_k"));
    check("form: zero parts rejected", from_form(std::nullopt, std::nullopt, {}).status == 400);
    const std::vector<std::string_view> two = {kJpeg, kPng};
    auto multi = from_form(std::nullopt, std::nullopt, two);
    check("form: two parts rejected", multi.status == 400 &&
                                          multi.error.find("exactly one") != std::string::npos);
    check("form: top_k 0 rejected", from_form(std::nullopt, std::string("0"), one).status == 400);
    check("form: top_k float rejected",
          from_form(std::nullopt, std::string("2.5"), one).status == 400);
    check("form: top_k text rejected",
          from_form(std::nullopt, std::string("abc"), one).status == 400);
    check("form: top_k 1000001 rejected",
          from_form(std::nullopt, std::string("1000001"), one).status == 400);
    const std::vector<std::string_view> gif = {"GIF89a"};
    check("form: GIF rejected", from_form(std::nullopt, std::nullopt, gif).status == 400);
}

void test_precheck_and_path() {
    check("precheck: no Content-Length is 411", precheck_content_length(false, 0) == 411);
    check("precheck: 1 KiB passes", precheck_content_length(true, 1024) == 0);
    check("precheck: exactly the limit passes",
          precheck_content_length(true, kMaxImageClassifyRequestBytes) == 0);
    check("precheck: limit + 1 is 413",
          precheck_content_length(true, kMaxImageClassifyRequestBytes + 1) == 413);

    for (const char* p : {"/api/v0/images/classify", "/api/v1/images/classify",
                          "/v0/images/classify", "/v1/images/classify"}) {
        check(std::string("path: matches ") + p, is_image_classify_path(p));
    }
    for (const char* p : {"/v1/images/classifyx", "/v1/classify", "/v1/images/edits",
                          "/v2/images/classify", "/images/classify"}) {
        check(std::string("path: ignores ") + p, !is_image_classify_path(p));
    }
}

}  // namespace

int main() {
    test_base64();
    test_sniff();
    test_json();
    test_form();
    test_precheck_and_path();
    if (failures == 0) {
        std::printf("\nAll image classify request checks passed.\n");
        return 0;
    }
    std::printf("\n%d image classify request check(s) failed.\n", failures);
    return 1;
}
