#include <caffe/caffe.hpp>
#include <iostream>
#include <string>

using namespace caffe;
using namespace std;

int main(int argc, char* argv[]) {
  // set cpu running software
  Caffe::set_mode(Caffe::CPU);

  // load net file, caffe::TEST
  Net<float> lenet(argv[1], caffe::TEST);

  // load net train file caffemodel
  lenet.CopyTrainedLayersFrom(argv[2]);

  /* Blob<type>* input_ptr = lenet.input_blobs()[0];
  input_ptr->Reshape(1, 1, 28, 28);

  Blob<type>* output_ptr = lenet.output_blobs()[0];
  output_ptr->Reshape(1, 10, 1, 1); */

  string network_name = lenet.name();
  cout << "network name : [" << network_name << "]" << endl;
  const vector<string>& layer_names = lenet.layer_names();
  cout << "network layers : " << layer_names.size() << endl;
  for (const auto s : layer_names)
    cout << "    " << s << endl;
  const vector<string>& blob_names = lenet.blob_names();
  cout << "network blobs : " << blob_names.size() << endl;
  for (const auto s : blob_names)
    cout << "    " << s << endl;

  return 0;
}
