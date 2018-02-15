# Tensorflow Libtoco

## //tensorflow/contrib/lite/utils:libtoco.so
- stores in the seperated directory `utils` that will avoid many git merge troubles
- TODOs
  + should have `libtoco` directoy its own
  + should have version to check
  + design library interface with namespace and header
- build
```sh
$ make build_libtoco
```
 
## sandbox/libtoco/
- `libtoco_test` to linked with `libtoco.so` and executed library functions
- TODOs
  + rewrite wiht cmake
  + some where in `libtoco.so` needs `libtensorflow_framework.so` which may be stripped
```sh
$ make build_libtoco_test
$ make run_libtoco
```

## another git repo
- TODOs
  + should have another git repo to keep track really things with `libtoco.so`
