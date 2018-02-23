#include <caffe/caffe.hpp>
#include <stdio.h>
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
    const vector<boost::shared_ptr<caffe::Blob<float> > >& blobs = layer->blobs();
    const caffe::LayerParameter& layer_param = layer->layer_param();
    int opcode_idx;

    tflite::BuiltinOperator builtin_code;
    string custom_code;

    // Generate unique key for the OperatorCode
    GetTFLiteBuiltinOrCustomCode(layer_param, builtin_code, custom_code);
    string map_key = EnumNameBuiltinOperator(builtin_code);
    if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
      map_key += '_';
      map_key += custom_code;
    }

    if (opcode_map.find(map_key) == opcode_map.end()) {
      opcode_map[map_key] = opcode_vector.size();
      if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
        opcode_vector.push_back(tflite::CreateOperatorCode(builder, builtin_code, builder.CreateString(custom_code)));
      }
      else {
        opcode_vector.push_back(tflite::CreateOperatorCode(builder, builtin_code));
      }
    }
    else {
      opcode_idx = opcode_map[map_key];
    }

#if 0
    printf("\n=== Layer name %s\n", s.c_str());
    if (layer_param.bottom_size() > 0) {
      for (int i = 0; i < layer_param.bottom_size(); i++) {
        printf("bottom -- %d\n", tensor_map[layer_param.bottom(i)]);
      }
    }

    if (layer_param.top_size() > 0) {
      for (int i = 0; i < layer_param.top_size(); i++) {
        printf("top -- %d\n", tensor_map[layer_param.top(i)]);
      }
    }
#endif
  }

  /* Build TFLite SubGraph */
  auto tensors = builder.CreateVector(tensor_vector);
  auto inputs = builder.CreateVector(net.input_blob_indices());
  auto outputs = builder.CreateVector(net.output_blob_indices());
  auto operators = builder.CreateVector(operator_vector); // TODO

  flatbuffers::Offset<tflite::SubGraph> subgraph_list[1];
  subgraph_list[0] = tflite::CreateSubGraph(builder, tensors, inputs, outputs, operators);
  auto subgraphs = builder.CreateVector(subgraph_list, 1);

  /* Build TFLite Model */
  auto buffers = builder.CreateVector(buffer_vector);
  auto opcodes = builder.CreateVector(opcode_vector); // TODO

  unsigned int version = 3;
  auto description = builder.CreateString("");
  auto model = tflite::CreateModel(builder, version, opcodes, subgraphs, description, buffers);
  builder.Finish(model, "TFL3");

  /* Export to file */
  char* buf = reinterpret_cast<char*>(builder.GetBufferPointer());
  size_t size = builder.GetSize();

  std::ofstream outfile(argv[3], std::ofstream::binary);
  outfile.write(buf, size);
  outfile.close();
}
