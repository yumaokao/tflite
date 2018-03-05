#include <caffe/caffe.hpp>
#include <stdio.h>
#include <map>
#include <vector>
#include <cassert>
#include <memory>
#include "tensorflow/contrib/lite/schema/schema_generated.h"

using namespace std;

#define EXIT_WITH_MSG(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__); exit(EXIT_FAILURE);

void GetTFLiteBuiltinOrCustomCode(const caffe::LayerParameter& layer_param, tflite::BuiltinOperator& builtin_code, string& custom_code)
{
  const string layer_type_str = layer_param.type();
  if (layer_type_str.compare("Convolution") == 0) {
    builtin_code = tflite::BuiltinOperator_CONV_2D;
  }
  else if (layer_type_str.compare("InnerProduct") == 0) {
    builtin_code = tflite::BuiltinOperator_FULLY_CONNECTED;
  }
  else if (layer_type_str.compare("Pooling") == 0) {
    assert(layer_param.has_pooling_param());
    const caffe::PoolingParameter& param = layer_param.pooling_param();
    assert(param.has_pool());
    const string pool_func_str = param.PoolMethod_Name(param.pool());
    if (pool_func_str.compare("MAX") == 0) {
      builtin_code = tflite::BuiltinOperator_MAX_POOL_2D;
    }
    else if (pool_func_str.compare("AVE") == 0) {
      builtin_code = tflite::BuiltinOperator_AVERAGE_POOL_2D;
    }
    else {
      EXIT_WITH_MSG("Unsupport pooling method: %s\n", pool_func_str.c_str());
    }
  }
  else if (layer_type_str.compare("ReLU") == 0) {
    builtin_code = tflite::BuiltinOperator_RELU;
  }
  else if (layer_type_str.compare("Softmax") == 0) {
    builtin_code = tflite::BuiltinOperator_SOFTMAX;
  }
  else if (layer_type_str.compare("Input") == 0) {
    builtin_code = tflite::BuiltinOperator_CUSTOM;
    custom_code = "Input";
  }
  else {
    EXIT_WITH_MSG("Unsupport layer type: %s\n", layer_type_str.c_str());
  }
}

bool GetTFLiteBuiltinOptions(flatbuffers::FlatBufferBuilder& builder, const caffe::LayerParameter& layer_param, const tflite::BuiltinOperator& op, tflite::BuiltinOptions& option_type, flatbuffers::Offset<void>& builtin_options)
{
  if ((op == tflite::BuiltinOperator_AVERAGE_POOL_2D) ||
    (op == tflite::BuiltinOperator_MAX_POOL_2D)) {
    assert(layer_param.has_pooling_param());
    const caffe::PoolingParameter& param = layer_param.pooling_param();
    tflite::Pool2DOptionsBuilder option_builder(builder);
    // Pad
    if (param.has_pad() || param.has_pad_h() || param.has_pad_w()) {
      // TODO
      option_builder.add_padding(tflite::Padding_SAME);
    }
    // Kernel
    if (param.has_kernel_size()) {
      option_builder.add_filter_height(param.kernel_size());
      option_builder.add_filter_width(param.kernel_size());
    }
    else {
      if (param.has_kernel_h()) {
        option_builder.add_filter_height(param.kernel_h());
      }
      if (param.has_kernel_w()) {
        option_builder.add_filter_width(param.kernel_w());
      }
    }
    // Stride
    if (param.has_stride()) {
      option_builder.add_stride_h(param.stride());
      option_builder.add_stride_w(param.stride());
    }
    else {
      if (param.has_stride_h()) {
        option_builder.add_stride_h(param.stride_h());
      }
      if (param.has_stride_w()) {
        option_builder.add_stride_w(param.stride_w());
      }
    }
    option_builder.add_fused_activation_function(tflite::ActivationFunctionType_NONE);
    builtin_options = option_builder.Finish().Union();
    option_type = tflite::BuiltinOptions_Pool2DOptions;
    return true;
  }
  else if (op == tflite::BuiltinOperator_CONV_2D) {
    assert(layer_param.has_convolution_param());
    const caffe::ConvolutionParameter& param = layer_param.convolution_param();
    tflite::Conv2DOptionsBuilder option_builder(builder);
    // Pad
    // TODO
    option_builder.add_padding(tflite::Padding_SAME);
    // Stride
    // TODO: stride_h and stride_w and multi-dimension stride
    assert(param.stride_size() == 1);
    option_builder.add_stride_h(param.stride(0));
    option_builder.add_stride_w(param.stride(0));
    option_builder.add_fused_activation_function(tflite::ActivationFunctionType_NONE);
    builtin_options = option_builder.Finish().Union();
    option_type = tflite::BuiltinOptions_Conv2DOptions;
    return true;
  }
  else if (op == tflite::BuiltinOperator_FULLY_CONNECTED) {
    tflite::FullyConnectedOptionsBuilder option_builder(builder);
    option_builder.add_fused_activation_function(tflite::ActivationFunctionType_NONE);
    builtin_options = option_builder.Finish().Union();
    option_type = tflite::BuiltinOptions_FullyConnectedOptions;
    return true;
  }
  else if (op == tflite::BuiltinOperator_RELU) {
    return false;
  }
  else if (op == tflite::BuiltinOperator_SOFTMAX) {
    tflite::SoftmaxOptionsBuilder option_builder(builder);
    option_builder.add_beta(1.0f);
    builtin_options = option_builder.Finish().Union();
    option_type = tflite::BuiltinOptions_SoftmaxOptions;
    return true;
  }
  else if (op == tflite::BuiltinOperator_CUSTOM) {
    // TODO
    return false;
  }
  else {
    EXIT_WITH_MSG("Unsupport tflite op: %s\n", tflite::EnumNameBuiltinOperator(op));
  }
}

int main(int argc, char* argv[]) {

  // load Net with caffe::TEST
  caffe::Net<float> net(argv[1], caffe::TEST);

  // load net train file caffemodel
  net.CopyTrainedLayersFrom(argv[2]);

  // TFLite builder
  flatbuffers::FlatBufferBuilder builder(1024);

  vector<flatbuffers::Offset<tflite::Buffer> > buffer_vector;
  vector<flatbuffers::Offset<tflite::Tensor> > tensor_vector;
  map<string, int> tensor_map;
  vector<flatbuffers::Offset<tflite::OperatorCode> > opcode_vector;
  map<string, int> opcode_map;
  vector<flatbuffers::Offset<tflite::Operator> > operator_vector;

  // TODO: Difference between boost::shared_ptr and std::shared_ptr

  /* Intermediate blobs (i.e. inputs/outputs of all the layers) */
  const vector<string>& net_blob_names = net.blob_names();
  const vector<boost::shared_ptr<caffe::Blob<float> > >& net_blobs = net.blobs();
  assert(net_blob_names.size() == net_blobs.size());
  for (auto idx = 0 ; idx < net_blobs.size() ; idx ++) {
    const string cur_blob_name = net_blob_names[idx];
    const boost::shared_ptr<caffe::Blob<float> > cur_blob = net_blobs[idx];

    const auto buffer_idx = buffer_vector.size();
    auto blob_shape = cur_blob->shape();
    // NCHW -> NHWC
    const auto shape = builder.CreateVector(vector<int>({blob_shape[0], blob_shape[2], blob_shape[3], blob_shape[1]}));
    const auto type = tflite::TensorType_FLOAT32;
    const auto name = builder.CreateString(cur_blob_name);

    tensor_map[cur_blob_name] = tensor_vector.size();
    tensor_vector.push_back(tflite::CreateTensor(builder, shape, type, buffer_idx, name));
    buffer_vector.push_back(tflite::CreateBuffer(builder));
  }

  /* Caffe Layers */
  const vector<string>& layer_names = net.layer_names();
  for (const auto s : layer_names) {
    const boost::shared_ptr<caffe::Layer<float> >& layer = net.layer_by_name(s);
    const vector<boost::shared_ptr<caffe::Blob<float> > >& layer_blobs = layer->blobs();
    const caffe::LayerParameter& layer_param = layer->layer_param();

    // Required information to create an Operator
    int opcode_idx;
    vector<int> inputs_vector;
    vector<int> outputs_vector;
    bool has_option;
    tflite::BuiltinOptions option_type;
    flatbuffers::Offset<void> builtin_options;

    // Generate unique key for the OperatorCode
    tflite::BuiltinOperator builtin_code;
    string custom_code;
    GetTFLiteBuiltinOrCustomCode(layer_param, builtin_code, custom_code);
    string map_key = tflite::EnumNameBuiltinOperator(builtin_code);
    if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
      map_key += '_';
      map_key += custom_code;
    }

    // Find opcode index or create opcode
    if (opcode_map.find(map_key) == opcode_map.end()) {
      opcode_map[map_key] = opcode_vector.size();
      if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
        opcode_vector.push_back(tflite::CreateOperatorCode(builder, builtin_code, builder.CreateString(custom_code)));
      }
      else {
        opcode_vector.push_back(tflite::CreateOperatorCode(builder, builtin_code));
      }
    }
    opcode_idx = opcode_map[map_key];

    // Find input/output index
    if (layer_param.bottom_size() > 0) {
      for (auto i = 0; i < layer_param.bottom_size(); i++) {
        inputs_vector.push_back(tensor_map[layer_param.bottom(i)]);
      }
    }

    if (layer_param.top_size() > 0) {
      for (auto i = 0; i < layer_param.top_size(); i++) {
        outputs_vector.push_back(tensor_map[layer_param.top(i)]);
      }
    }

    // Parameter blobs (needs to be in `inputs_vector` and `tensor_vector` as well)
    if ((builtin_code == tflite::BuiltinOperator_CONV_2D) ||
      (builtin_code == tflite::BuiltinOperator_FULLY_CONNECTED)) {
      for (int i = 0; i < layer_blobs.size(); i++) {

        const boost::shared_ptr<caffe::Blob<float> > cur_blob = layer_blobs[i];
        vector<int> blob_shape = cur_blob->shape();
        size_t buffer_count = cur_blob->count();
        const float* buffer_content = cur_blob->cpu_data();
        flatbuffers::Offset<tflite::Buffer> buffer;

        if (blob_shape.size() == 4) {
          // NCHW -> NHWC
          unique_ptr<float[]> reorder_data(new float[buffer_count]);

          int n_count = blob_shape[0];
          int c_count = blob_shape[1];
          int h_count = blob_shape[2];
          int w_count = blob_shape[3];
          for (auto n = 0 ; n < n_count ; n ++) {
            for (auto c = 0 ; c < c_count ; c ++) {
              for (auto hw = 0 ; hw < h_count * w_count ; hw ++) {
                int before_ofst = (n * c_count + c) * h_count * w_count + hw;
                int after_ofst = (n * h_count * w_count + hw) * c_count + c;
                reorder_data[after_ofst] = buffer_content[before_ofst];
              }
            }
          }
          buffer = tflite::CreateBuffer(builder,
              builder.CreateVector(reinterpret_cast<uint8_t*>(reorder_data.get()), buffer_count * sizeof(float)));
          blob_shape = {n_count, h_count, w_count, c_count};
        }
        else {
          buffer = tflite::CreateBuffer(builder,
              builder.CreateVector(reinterpret_cast<const uint8_t*>(buffer_content), buffer_count * sizeof(float)));
        }

        int tensor_idx = tensor_vector.size();
        int buffer_idx = buffer_vector.size();
        const auto type = tflite::TensorType_FLOAT32;
        // TODO: proper name for the blob buffer
        string name = s + "_param_blob_" + std::to_string(i);
        inputs_vector.push_back(tensor_idx);
        tensor_vector.push_back(tflite::CreateTensor(builder, builder.CreateVector(blob_shape),
                                                    type, buffer_idx, builder.CreateString(name)));
        buffer_vector.push_back(buffer);
      }
    }


    // Get the op option
    // TODO: custom options?
    has_option = GetTFLiteBuiltinOptions(builder, layer_param, builtin_code, option_type, builtin_options);

    if (has_option) {
      operator_vector.push_back(tflite::CreateOperator(builder, opcode_idx,
                                          builder.CreateVector(inputs_vector),
                                          builder.CreateVector(outputs_vector),
                                          option_type, builtin_options));
    }
    else {
      operator_vector.push_back(tflite::CreateOperator(builder, opcode_idx,
                                          builder.CreateVector(inputs_vector),
                                          builder.CreateVector(outputs_vector)));
    }
  }

  /* Build TFLite SubGraph */
  auto tensors = builder.CreateVector(tensor_vector);
  auto inputs = builder.CreateVector(net.input_blob_indices());
  auto outputs = builder.CreateVector(net.output_blob_indices());
  auto operators = builder.CreateVector(operator_vector);

  auto subgraph = tflite::CreateSubGraph(builder,
                              builder.CreateVector(tensor_vector),
                              builder.CreateVector(net.input_blob_indices()),
                              builder.CreateVector(net.output_blob_indices()),
                              builder.CreateVector(operator_vector));

  /* Build TFLite Model */
  unsigned int version = 3;
  auto description = builder.CreateString("");
  auto model = tflite::CreateModel(builder, version,
                    builder.CreateVector(opcode_vector),
                    builder.CreateVector(&subgraph, 1),
                    description, builder.CreateVector(buffer_vector));
  builder.Finish(model, "TFL3");

  /* Export to file */
  char* buf = reinterpret_cast<char*>(builder.GetBufferPointer());
  size_t size = builder.GetSize();

  std::ofstream outfile(argv[3], std::ofstream::binary);
  outfile.write(buf, size);
  outfile.close();
}
