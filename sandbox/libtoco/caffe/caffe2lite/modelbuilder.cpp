#include <iostream>
#include <fstream>
#include <string>

#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/version.h"

namespace tflite {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;

class TfLiteFlatbufferModelBuilder {
 public:
  TfLiteFlatbufferModelBuilder() {
    buffers_.push_back(
        CreateBuffer(builder_, builder_.CreateVector(std::vector<uint8_t>{})));
  }

  TfLiteFlatbufferModelBuilder(const std::vector<BuiltinOperator>& builtin_ops,
                               const std::vector<std::string>& custom_ops) {
    buffers_.push_back(
        CreateBuffer(builder_, builder_.CreateVector(std::vector<uint8_t>{})));

    /* for (const auto& iter : builtin_ops) {
      resolver_.AddBuiltin(iter, &fake_op_);
    }
    for (const auto& iter : custom_ops) {
      resolver_.AddCustom(iter.data(), &fake_op_);
    } */
  }

  void AddTensor(const std::vector<int>& shape, tflite::TensorType type,
                 const std::vector<uint8_t>& buffer, const char* name) {
    int buffer_index = 0;
    if (!buffer.empty()) {
      buffer_index = buffers_.size();
      buffers_.push_back(CreateBuffer(builder_, builder_.CreateVector(buffer)));
    }
    tensors_.push_back(CreateTensorDirect(builder_, &shape, type, buffer_index,
                                          name, /*quantization=*/0));
  }

  void AddOperator(const std::vector<int32_t>& inputs,
                   const std::vector<int32_t>& outputs,
                   tflite::BuiltinOperator builtin_op, const char* custom_op) {
    operator_codes_.push_back(
        CreateOperatorCodeDirect(builder_, builtin_op, custom_op));
    operators_.push_back(CreateOperator(
        builder_, operator_codes_.size() - 1, builder_.CreateVector(inputs),
        builder_.CreateVector(outputs), BuiltinOptions_NONE,
        /*builtin_options=*/0,
        /*custom_options=*/0, tflite::CustomOptionsFormat_FLEXBUFFERS));
  }

  void FinishModel(const std::vector<int32_t>& inputs,
                   const std::vector<int32_t>& outputs) {
    auto subgraph = std::vector<Offset<SubGraph>>({CreateSubGraph(
        builder_, builder_.CreateVector(tensors_),
        builder_.CreateVector(inputs), builder_.CreateVector(outputs),
        builder_.CreateVector(operators_),
        builder_.CreateString("test_subgraph"))});
    auto result = CreateModel(
        builder_, TFLITE_SCHEMA_VERSION, builder_.CreateVector(operator_codes_),
        builder_.CreateVector(subgraph), builder_.CreateString("test_model"),
        builder_.CreateVector(buffers_));
    tflite::FinishModelBuffer(builder_, result);
  }

 private:
  FlatBufferBuilder builder_;
  // MutableOpResolver resolver_;
  // TfLiteRegistration fake_op_;
  std::vector<Offset<Operator>> operators_;
  std::vector<Offset<OperatorCode>> operator_codes_;
  std::vector<Offset<Tensor>> tensors_;
  std::vector<Offset<Buffer>> buffers_;
};
} //namespace tflite

int main(int argc, char* argv[]) {
  std::cout << "YMK hello" << std::endl;
}
