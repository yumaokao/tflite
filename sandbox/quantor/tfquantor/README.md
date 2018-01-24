# Tensorflow Quantor Module

## tfquantor/quantize
- replica of `tensorflow/contrib/quantize`
- replace all tensorflow.contrib.quantize
  `from tensorflow.contrib.quantize.python import xxx`
  to
  `from tfquantor.quantize import xxx`
- comment out `@ops.RegisterGradient('FoldFusedBatchNormGrad')`
  since tensorflow.contrib.quantize already register one
 
## install directly with pip
```sh
$ make pip_install_user
$ pip install --upgrade --no-deps --force-reinstall --user .
```
