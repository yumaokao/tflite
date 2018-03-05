/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <iostream>
#include <vector>

#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"

int main(int argc, char** argv) {
  toco::Model* model = new toco::Model();
  if (model == nullptr) {
    std::cout << "Create toco::Model instance failed." << std::endl;
    return -1;
  }

  auto& array = model->GetOrCreateArray("test");
  array.data_type = toco::ArrayDataType::kInt32;
  auto array_dim = array.mutable_shape()->mutable_dims();
  *array_dim = std::vector<int>(1, 4);

  FixNoMissingArray(model);

  std::cout << "Create toco::Model instance success." << std::endl;
}
