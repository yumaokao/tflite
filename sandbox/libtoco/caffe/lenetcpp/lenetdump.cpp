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
  for (const auto s : layer_names) {
    const boost::shared_ptr<Layer<float> >& layer = lenet.layer_by_name(s);
    const vector<boost::shared_ptr<Blob<float> > >& blobs = layer->blobs();
    const LayerParameter& layer_param = layer->layer_param();
    cout << "  " << s << ": " << layer_param.type()
         << ", parameter blobs " << blobs.size()
         // << ", blobs " << layer_param.blobs_size()
         << ", bottoms " << layer_param.bottom_size()
         << ", tops " << layer_param.top_size() << endl;
    // parameter blobs
    if (blobs.size() > 0) {
      cout << "    [parameter blobs]" << endl;
      for (int i = 0; i < blobs.size(); i++) {
        const boost::shared_ptr<Blob<float> > b = blobs[i];
        cout << "      blob[" << i << "]: " << b->shape_string() << endl;
        // TODO(yumaokao): conv weights in kOIHW order,
        //                 however kHWIO in Tensorflow, kOHWI in TFlite.

        // dump with cpu_data(), since read only
        const float* bdata = b->cpu_data();
        cout << "      ->data:";
        for (int i = 0; i < b->count(); i++) {
          cout << " " << bdata[i];
          if ( i > 8)
            break;
        }
        cout << endl;
      }
    }
    // has_convolution_param
    if (layer_param.has_convolution_param()) {
      const ConvolutionParameter& param = layer_param.convolution_param();
      cout << "    [convolution_param]" << endl;
      if (param.has_num_output())
        cout << "      num_output: " << param.num_output() << endl;
      if (param.has_bias_term())
        cout << "      bias_term: " << param.bias_term() << endl;
      cout << "      pad_size: " << param.pad_size() << endl;
      cout << "      kernel_size: " << param.kernel_size_size()
           << " (" << param.kernel_size(0) << ")" << endl;
      cout << "      stride_size: " << param.stride_size()
           << " (" << param.stride(0) << ")" << endl;
    }

    // has_pooling_param
    if (layer_param.has_pooling_param()) {
      const PoolingParameter& param = layer_param.pooling_param();
      cout << "    [pooling_param]" << endl;
      if (param.has_pool())
        cout << "      pool method: "
             << param.PoolMethod_Name(param.pool()) << endl;
      if (param.has_kernel_size())
        cout << "      kernel size: " << param.kernel_size() << endl;
      if (param.has_pad())
        cout << "      pad: " << param.pad() << endl;
      if (param.has_stride())
        cout << "      stride: " << param.stride() << endl;
    }

    // has_relu_param
    if (layer_param.has_relu_param()) {
      const ReLUParameter& param = layer_param.relu_param();
    }

    // has_inner_product_param
    if (layer_param.has_inner_product_param()) {
      const InnerProductParameter& param = layer_param.inner_product_param();
      cout << "    [inner_product_param]" << endl;
      if (param.has_num_output())
        cout << "      num_output: " << param.num_output() << endl;
      if (param.has_bias_term())
        cout << "      bias_term: " << param.bias_term() << endl;
    }
  }
  const vector<string>& blob_names = lenet.blob_names();
  cout << "network blobs : " << blob_names.size() << endl;
  for (const auto s : blob_names)
    cout << "    " << s << endl;

  return 0;
}
