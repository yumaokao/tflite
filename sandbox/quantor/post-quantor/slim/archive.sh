ORG_DIR=${1%/}
DST_DIR=${2%/}

if [ -d "$DST_DIR" ]; then 
  echo "Destination directory exist!"
  exit 1
fi

# TensorFlow FP32
mkdir -p $DST_DIR/TensorFlow/FP32/golden
cp $ORG_DIR/frozen_$ORG_DIR.pb $DST_DIR/TensorFlow/FP32/model.pb
if [ $? -ne 0 ]; then
  rm -rf $DST_DIR
  exit 1
fi
cp $ORG_DIR/dump_frozen_jpeg/golden/* $DST_DIR/TensorFlow/FP32/golden/
if [ $? -ne 0 ]; then
  rm -rf $DST_DIR
  exit 1
fi

# TensorFlow FakeQuant
mkdir -p $DST_DIR/TensorFlow/FakeQuant/golden
cp $ORG_DIR/quantor/frozen.pb $DST_DIR/TensorFlow/FakeQuant/fakequant_model.pb
if [ $? -ne 0 ]; then
  rm -rf $DST_DIR
  exit 1
fi
cp $ORG_DIR/quantor/dump_frozen_jpeg/golden/* $DST_DIR/TensorFlow/FakeQuant/golden/
if [ $? -ne 0 ]; then
  rm -rf $DST_DIR
  exit 1
fi

# TFLite FP32
mkdir -p $DST_DIR/TFLite/FP32/golden
cp $ORG_DIR/float_model.lite $DST_DIR/TFLite/FP32/float_model.lite
if [ $? -ne 0 ]; then
  rm -rf $DST_DIR
  exit 1
fi
cp $ORG_DIR/dots/toco_AFTER_TRANSFORMATIONS.dot.pdf $DST_DIR/TFLite/FP32/
if [ $? -ne 0 ]; then
  rm -rf $DST_DIR
  exit 1
fi
cp $ORG_DIR/dump_tflite_jpeg/golden/* $DST_DIR/TFLite/FP32/golden/
if [ $? -ne 0 ]; then
  rm -rf $DST_DIR
  exit 1
fi

# TFLite UINT8
mkdir -p $DST_DIR/TFLite/UINT8/golden
cp $ORG_DIR/quantor/model.lite $DST_DIR/TFLite/UINT8/uint8_model.lite
if [ $? -ne 0 ]; then
  rm -rf $DST_DIR
  exit 1
fi
cp $ORG_DIR/quantor/dots/toco_AFTER_TRANSFORMATIONS.dot.pdf $DST_DIR/TFLite/UINT8/
if [ $? -ne 0 ]; then
  rm -rf $DST_DIR
  exit 1
fi
cp $ORG_DIR/quantor/dump_tflite_jpeg/golden/* $DST_DIR/TFLite/UINT8/golden/
if [ $? -ne 0 ]; then
  rm -rf $DST_DIR
  exit 1
fi

# Compress and pack
tar zcvf $DST_DIR.tar.gz $DST_DIR
rm -rf $DST_DIR
