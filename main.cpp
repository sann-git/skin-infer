// skin-infer: minimal CLI ONNX Runtime inference for the KararSkin
// ViT-Base skin-type / acne classifiers.
//
// Usage: ./skin-infer <model.onnx> <image.jpg>
// Prints: {"scores":[0.1,0.2,0.7]}   (softmax probabilities, in id2label order)
//
// Preprocessing matches the ViTImageProcessor config exactly:
//   resize -> 224x224 (bilinear)
//   rescale by 1/255
//   normalize: (x - 0.5) / 0.5   per channel
//   layout: NCHW, float32

#include <onnxruntime_cxx_api.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

static const int IMG_SIZE = 224;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.onnx> <image.jpg>\n", argv[0]);
        return 2;
    }
    const char* model_path = argv[1];
    const char* image_path = argv[2];

    // ---- Load + decode image ----
    int w, h, channels;
    unsigned char* img = stbi_load(image_path, &w, &h, &channels, 3); // force RGB
    if (!img) {
        fprintf(stderr, "{\"error\":\"failed to load image: %s\"}\n", image_path);
        return 1;
    }

    // ---- Resize to 224x224 ----
    std::vector<unsigned char> resized(IMG_SIZE * IMG_SIZE * 3);
    stbir_resize_uint8_linear(
        img, w, h, 0,
        resized.data(), IMG_SIZE, IMG_SIZE, 0,
        STBIR_RGB
    );
    stbi_image_free(img);

    // ---- Rescale (1/255) + normalize ((x-0.5)/0.5), pack as NCHW float32 ----
    std::vector<float> input_tensor(3 * IMG_SIZE * IMG_SIZE);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                unsigned char px = resized[(y * IMG_SIZE + x) * 3 + c];
                float v = (px / 255.0f - 0.5f) / 0.5f;
                input_tensor[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = v;
            }
        }
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "skin-infer");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, model_path, opts);

        Ort::AllocatorWithDefaultOptions allocator;

        // Assume single input, named whatever it's named (grab it dynamically)
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        std::vector<const char*> input_names = { input_name.get() };
        std::vector<const char*> output_names = { output_name.get() };

        std::array<int64_t, 4> input_shape = { 1, 3, IMG_SIZE, IMG_SIZE };

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            mem_info, input_tensor.data(), input_tensor.size(),
            input_shape.data(), input_shape.size()
        );

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(), &input, 1,
            output_names.data(), 1
        );

        float* logits = output_tensors[0].GetTensorMutableData<float>();
        auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t num_classes = out_shape.back();

        // softmax
        float maxv = *std::max_element(logits, logits + num_classes);
        std::vector<float> exps(num_classes);
        float sum = 0.0f;
        for (int64_t i = 0; i < num_classes; i++) {
            exps[i] = std::exp(logits[i] - maxv);
            sum += exps[i];
        }

        printf("{\"scores\":[");
        for (int64_t i = 0; i < num_classes; i++) {
            printf("%s%.6f", (i ? "," : ""), exps[i] / sum);
        }
        printf("]}\n");

    } catch (const Ort::Exception& e) {
        fprintf(stderr, "{\"error\":\"onnxruntime: %s\"}\n", e.what());
        return 1;
    }

    return 0;
}
