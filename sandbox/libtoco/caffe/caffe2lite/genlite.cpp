#include <iostream>
#include <fstream>
#include <string>

#include "tensorflow/contrib/lite/schema/schema_generated.h"

using namespace tflite;

int main(int argc, char* argv[]) {

  const uint16_t batch_count = 1;
  const uint16_t src_height = 3;
  const uint16_t src_width = 3;
  const uint16_t src_channel = 1;
  const uint16_t new_batch = 1;
  const uint16_t new_height = 1;
  const uint16_t new_width = 1;
  const uint16_t new_channel = 9;

  flatbuffers::FlatBufferBuilder builder(1024);

  /* OP Code */
  flatbuffers::Offset<OperatorCode> opcode_list[1];
  opcode_list[0] = CreateOperatorCode(builder, BuiltinOperator::BuiltinOperator_RESHAPE);
  auto operator_codes = builder.CreateVector(opcode_list, 1);

  /* Buffer */
  flatbuffers::Offset<Buffer> buffer_list[2];
  buffer_list[0] = CreateBuffer(builder);
  buffer_list[1] = CreateBuffer(builder);
  auto buffers = builder.CreateVector(buffer_list, 2);

  /* Operator */
  flatbuffers::Offset<Operator> operator_list[1];

  int input_list[1] = {0};
  int output_list[1] = {1};
  auto inputs = builder.CreateVector(input_list, 1);
  auto outputs = builder.CreateVector(output_list, 1);
  int new_shape_list[4] = {new_batch, new_height, new_width, new_channel};
  auto new_shape = builder.CreateVector(new_shape_list, 4);
  auto builtin_options = CreateReshapeOptions(builder, new_shape).Union();
  operator_list[0] = CreateOperator(builder, 0, inputs, outputs, BuiltinOptions_ReshapeOptions, builtin_options);
  auto operators = builder.CreateVector(operator_list, 1);

  /* Tensor */
  flatbuffers::Offset<Tensor> tensor_list[2];
  int input_shape_list[] = {batch_count, src_height, src_width, src_channel};
  int output_shape_list[] = {new_batch, new_height, new_width, new_channel};
  auto input_shape = builder.CreateVector(input_shape_list, 4);
  auto output_shape = builder.CreateVector(output_shape_list, 4);

  auto input_type = TensorType_FLOAT32;
  auto output_type = TensorType_FLOAT32;
  auto input_name = builder.CreateString("input");
  auto output_name = builder.CreateString("output");
  tensor_list[0] = CreateTensor(builder, input_shape, input_type, 0, input_name);
  tensor_list[1] = CreateTensor(builder, output_shape, output_type, 1, output_name);
  auto tensors = builder.CreateVector(tensor_list, 2);

  /* SubGraph */
  flatbuffers::Offset<SubGraph> subgraph_list[1];
  subgraph_list[0] = CreateSubGraph(builder, tensors, inputs, outputs, operators);
  auto subgraphs = builder.CreateVector(subgraph_list, 1);

  /* Model */
  unsigned int version = 3;
  auto description = builder.CreateString("");
  auto model = CreateModel(builder, version, operator_codes, subgraphs, description, buffers);

  builder.Finish(model, "TFL3");
  char* buf = reinterpret_cast<char*>(builder.GetBufferPointer());
  size_t size = builder.GetSize();

  std::ofstream outfile(argv[1], std::ofstream::binary);
  outfile.write(buf, size);
  outfile.close();
}
