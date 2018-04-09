import array_extra_info_pb2
import types_pb2


def main():
  arrayextrainfo = array_extra_info_pb2.ArraysExtraInfo()
  entry = arrayextrainfo.entries.add()
  entry.name = 'YMK0'
  entry.min = 0.0
  entry.max = 6.0
  entry.data_type = types_pb2.QUANTIZED_UINT8
  entry = arrayextrainfo.entries.add()
  entry.name = 'YMK1'
  entry.min = -6.0
  entry.max = 10.0
  entry.data_type = types_pb2.QUANTIZED_UINT8

  binstr = arrayextrainfo.SerializeToString()
  print(binstr)
  print()

  arrayextrainfo_r = array_extra_info_pb2.ArraysExtraInfo()
  arrayextrainfo_r.ParseFromString(binstr)
  print(arrayextrainfo_r)


if __name__ == '__main__':
  main()
