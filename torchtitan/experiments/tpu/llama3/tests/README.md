# Running Llama 3 tests on Forge

Here is an example invocation of running llama3 tests on all devices:

```
blaze test -c opt --config=cuda //third_party/py/torchtitan/experiments/tpu/llama3/tests:test_llama3_all
```

Notes:

- Running on GPU requires compilation with CUDA (--config=cuda).
  It's safe to run all tests with the cuda config.
- Be sure to run with "-c opt" to avoid Forge OOMs during build.

## Available Targets

The following test targets are available to test a single backend:

- All tests: `test_llama3_all`
- GPU: `test_llama3_gpu` (equivalent to blaze tag "requires-gpu-nvidia")

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

- TPU: `test_llama3_tpu_vl` (runs on a 1x1 viperlite TPU)
  
  Specify `tpu_{tpu_name}` after the test name. Check
  `third_party/py/torchtitan/experiments/tpu/py3.bzl` for available TPUs.

- CPU: `test_llama3`