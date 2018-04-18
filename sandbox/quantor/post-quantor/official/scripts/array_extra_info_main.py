import array_extra_info_pb2
import types_pb2


def main():
  arrayextrainfo = array_extra_info_pb2.ArraysExtraInfo()
  entry = arrayextrainfo.entries.add()
  entry.name = 'array0'
  entry.min = 0.0
  entry.max = 6.0
  entry.data_type = types_pb2.QUANTIZED_UINT8
  entry = arrayextrainfo.entries.add()
  entry.name = 'array1'
  entry.min = -6.0
  entry.max = 10.0
  entry.data_type = types_pb2.QUANTIZED_UINT8

  # binstr = arrayextrainfo.SerializeToString()
  # print(binstr)
  # print()

  with open('array_extra_info.pb', 'wb') as f:
      f.write(arrayextrainfo.SerializeToString())

  arrayextrainfo_r = array_extra_info_pb2.ArraysExtraInfo()
  with open('array_extra_info.pb', 'rb') as f:
    arrayextrainfo_r.ParseFromString(f.read())

  # arrayextrainfo_r.ParseFromString(binstr)
  print(arrayextrainfo_r)


if __name__ == '__main__':
  main()
