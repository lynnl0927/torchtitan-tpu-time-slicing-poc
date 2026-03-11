# Running Qwen3 tests on Forge

Here is an example invocation of running Qwen3 tests on all devices:

```
blaze test -c opt --config=cuda //third_party/py/torchtitan/experiments/tpu/qwen3/tests:test_qwen3_all
```

Notes:

- Running on GPU requires compilation with CUDA (--config=cuda).
  It's safe to run all tests with the cuda config.
- Be sure to run with "-c opt" to avoid Forge OOMs during build.

The following test targets are available:

- All tests: `test_qwen3_all`
- GPU: `test_qwen3_gpu` (equivalent to blaze tag "requires-gpu-nvidia")

  Targets are also available for specific GPUs or multi-GPU

  ```
  # Run a test case on 1 Nvidia H100 GPUs:
  py3_test(name="my_library_test",
          run_on_gpu_h100 = True,
          ...)

  # Run a test case on 2 Nvidia H100 GPUs:
  py3_test(name="my_library_test",
            run_on_gpu_h100x2 = True,
            ...)
  ```

- TPU: `test_qwen3_tpu_vl` (runs on a 1x1 viperlite TPU)

  Specify `tpu_{tpu_name}` after the test name. Check
  `third_party/py/torchtitan/experiments/tpu/py3.bzl` for available TPUs.

- CPU: `test_qwen3`