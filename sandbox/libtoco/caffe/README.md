# Caffe

## Prepare Building Env.
- for Arch Linux, just install caffe
```sh
$ yaourt caffe-cpu
```
 
## Download and Configure
- clone and Makefile.config
```sh
$ git clone https://github.com/BVLC/caffe.git
$ cd caffe && cp Makefile.config.example Makefile.config
  # uses these to make building simpler
  CPU_ONLY := 1
  USE_OPENCV := 0
  # AUR caffe-cpu uses openblas
  BLAS := open
```

## Build
```sh
$ make all
# $ make test
# $ make runtest
$ make pycaffe
```

## MNIST
- ref: http://caffe.berkeleyvision.org/gathered/examples/mnist.html

- Prepare Datasets
```sh
$ ./data/mnist/get_mnist.sh
$ ./examples/mnist/create_mnist.sh
```

## LeNet
- Train
```sh
$ vim examples/mnist/lenet_solver.prototxt
  # solver mode: CPU or GPU
  solver_mode: CPU
$ ./examples/mnist/train_lenet.sh
$ ls -lh ./examples/mnist/lenet_iter_*
examples/mnist/lenet_iter_10000.caffemodel
examples/mnist/lenet_iter_10000.solverstate
```

- lenetcpp
```sh
$ cd caffe/lenetcpp
$ mkdir build && cd build
$ cmake ../
$ make get_lenet_model
$ make run_lenetinf
  res: 1 0.999654
```
