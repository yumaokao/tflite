.PHONY: all clean
.PHONY: spliter
.PHONY: array_extra_info


all:
	@ echo "all models"

spliter:
	@ python spliter.py

clean:
	@ rm -rf array_extra_info_pb2.py types_pb2.py __pycache__

array_extra_info:
	@ protoc array_extra_info.proto --python_out=./
	@ protoc types.proto --python_out=./
	@ python array_extra_info_main.py
