#include <caffe/caffe.hpp>
#include "tensorflow/contrib/lite/schema/schema_generated.h"

using namespace std;

int main(int argc, char* argv[]) {

  // load Net with caffe::TEST
  caffe::Net<float> net(argv[1], caffe::TEST);

  // load net train file caffemodel
  net.CopyTrainedLayersFrom(argv[2]);

  // TFLite builder
  flatbuffers::FlatBufferBuilder builder(1024);

  vector<flatbuffers::Offset<tflite::Buffer> > buffer_vector;
  vector<flatbuffers::Offset<tflite::Tensor> > tensor_vector;
  vector<flatbuffers::Offset<tflite::OperatorCode> > opcode_vector;
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

    tensor_vector.push_back(tflite::CreateTensor(builder, shape, type, buffer_idx, name));
    buffer_vector.push_back(tflite::CreateBuffer(builder));
  }

  /* Caffe Layers */

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
