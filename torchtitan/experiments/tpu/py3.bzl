"""Build extensions for Python with type checking, strict dependencies and tests
on GPUs/TPUs.

This is forked from go/py3-bzl, adding support for:
- Additional blaze tags for each device type.
- Additional targets for specific GPU types and multi-GPU.

The py3_test def makes it easy to run the test cases on GPU or TPU using Forge.
The number of GPU/TPU devices on Forge is limited. If your tests takes very long
check the queues states on go/forge-for-ml and consider switching accelerators.

Example
load("//torchtitan/experiments/tpu:py3.bzl", "py3_library", "py3_test")
# Define a library with struct dependency and pytype checks.
py3_library(name="my_library", ...)

# Run a test case on CPU, GPU and TPU:
py3_test(name="my_library_test",
          run_on_gpu = True, # This defaults to any available GPU, i.e "requires-gpu-nvidia"
          run_on_df = True, ...)

# Run a test case on 1 Nvidia H100 GPUs:
py3_test(name="my_library_test",
          run_on_gpu_h100 = True,
          ...)

# Run a test case on 2 Nvidia H100 GPUs:
py3_test(name="my_library_test",
          run_on_gpu_h100x2 = True,
          ...)

For JAX users:
1) If you only need multiple devices but no GPU/TPU you can simulate multiple
XLA CPU devices we recommend using chex:
import chex
def setUpModule():
  chex.set_n_cpu_devices(4)
  # jax.local_devices(backend="cpu")) will return 4 devices.
  # On GPU and TPU JAX will still use the GPU/TPU devices by default.

2) It is possible to run multi-process tests on TPUs. You can use the main()
function provided in
google3/learning/brain/research/jax/tests/tpu/multiprocess_tpu_test.py.
"""

load(
    "//devtools/build_cleaner/skylark:build_defs.bzl",
    "register_extension_info",
)
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

def py3_binary(name = None, **kwargs):
    # For a much faster linking time.
    # Context: b/350522609 and go/be#py_binary.linking_mode.
    linking_mode = kwargs.pop("linking_mode", "sharded_dynamic")
    pytype_strict_binary(name = name, srcs_version = "PY3", python_version = "PY3", linking_mode = linking_mode, **kwargs)

def py3_library(name = None, **kwargs):
    pytype_strict_library(name = name, srcs_version = "PY3", **kwargs)

def py3_test(
        name = None,
        deps = [],
        tags = [],
        tags_cpu = [],
        tags_gpu = [],
        tags_tpu = [],
        args = [],
        args_cpu = [],
        args_gpu = [],
        args_tpu = [],
        args_tpu_megacore = [],
        main = None,
        # We do not default to generating a CPU test target, specifying a backend is required.
        run_on_cpu = False,
        run_on_gpu = False,
        run_on_gpu_nvidiax2 = False,
        run_on_gpu_a100 = False,
        run_on_gpu_h100 = False,
        run_on_gpu_h100x2 = False,
        run_on_jf = False,
        run_on_df = False,
        run_on_df_2x2 = False,
        run_on_pl = False,
        run_on_pf = False,
        run_on_pf_megacore = False,
        run_on_pf_2x2x1 = False,
        run_on_pf_2x2x1_megacore = False,
        run_on_vl = False,
        run_on_vl_4x2x1 = False,
        run_on_vf_megachip = False,
        run_on_vf_2x2x1_megachip = False,
        run_on_glp_1x1 = False,
        run_on_glp_2x4 = False,
        run_on_gf_1x1x1 = False,
        run_on_gf_2x2x1 = False,
        hermetic = True,
        pytype = True,
        device_flag = None,
        **kwargs):
    """Runs a py_test on multiple devices.

    Args:
        name: Test target name to generate suffixed with `test`.
        deps: Additional dependencies for the test targets.
        tags: Tags to be assigned to the different test targets.
        tags_cpu: Tags passed to the test targets running on CPU (after `tags`).
        tags_gpu: Tags passed to the test targets running on GPU (after `tags`).
        tags_tpu: Tags passed to the test targets running on TPU (after `tags`).
        args: Args passed to the different test targets.
        args_cpu: Args passed to the test targets running on CPU (after `args`).
        args_gpu: Args passed to the test targets running on GPU (after `args`).
        args_tpu: Args passed to the test targets running on TPU (after `args`).
        args_tpu_megacore: Args passed to the test targets running on TPU (after `args_tpu`).
        main: Optional main script to be run for the test.
        run_on_cpu: Run as normally on CPU.
        run_on_gpu: Also run the test on a machine with any Nvidia GPU.
        run_on_gpu_nvidiax2: Also run the test on a machine with 2 P100 or V100 GPUs.
        run_on_gpu_a100: Also run the test on a machine with 1 Nvidia A100 GPU.
        run_on_gpu_h100: Also run the test on a machine with 1 Nvidia H100 GPU.
        run_on_gpu_h100x2: Also run the test on a machine with 2 Nvidia H100 GPUs.
        run_on_jf: Also run on TPU JellyFish 1x1 (single chip, 2 cores).
        run_on_df: Also run on TPU DragonFish 1x1 (single chip, 2 cores).
        run_on_df_2x2: Also run on TPU DragonFish 2x2 (four chips, 8 cores),
        run_on_pl: Also run on TPU PufferLite 1x1 (single chip, 1 core).
        run_on_pf: Also run on TPU PufferFish 1x1 (single chip, 2 cores).
        run_on_pf_megacore: Also run on TPU PufferFish Megacore 1x1 (single chip, 1 core).
        run_on_pf_2x2x1: Also run on TPU Pufferfish 2x2x1 (4 devices, 8 cores).
        run_on_pf_2x2x1_megacore: Also run on TPU Pufferfish Megacore 2x2x1 (4 devices, 4 cores).
        run_on_vl: Also run on TPU Viperlite 1x1 (single chip, 1 core).
        run_on_vl_4x2x1: Also run on TPU Viperlite 4x2x1 (8 devices, 8 cores).
        run_on_vf_megachip: Also run on TPU Viperfish 1x1 (single chip, 1 TC + 2 SCs).
        run_on_vf_2x2x1_megachip: Also run on TPU Viperfish 2x2x1 (4 devices, 4 TCs + 8 SCs).
        run_on_glp_1x1: Also run on TPU Ghostlite 1x1 (1 chip, 1 core).
        run_on_glp_2x4: Also run on TPU Ghostlite 2x4 (8 chips, 8 cores).
        run_on_gf_1x1x1: Also run on TPU Ghostfish 1x1x1 (1 chip, 2 cores).
        run_on_gf_2x2x1: Also run on TPU Ghostfish 2x2x1 (4 chips, 8 cores).
        hermetic: If True run the CPU test as hermetic test (disallows
            non-local network access).
        pytype: Whether to perform type checking.
        device_flag: If non-empty, pass the device name (eg `pf` or `pl` or `df_2x2`) to the
          test binary via a flag called `device_flag`, e.g., `device_flag="device"` will run the
          test binary with --device=....
        **kwargs: Extra keyword arguments to the test.
    """
    if main == None:
        main = name + ".py"
    kwargs["main"] = main
    kwargs["srcs_version"] = "PY3"
    kwargs["python_version"] = "PY3"

    if hermetic:
        # Taken from google3/production/dependency/rpc/testing/hermetic/build_defs.bzl.
        allow_network_tag = "requires-net:external"
        if allow_network_tag in tags:
            fullname = "//" + native.package_name() + ":" + name
            fail("Hermetic test '%s' is not allowed to request external network. " % fullname +
                 "Please, remove '%s' from tags: %s." % (allow_network_tag, ", ".join(tags)))
        tags = tags + ["requires-net:loopback"]

    py_test_macro = pytype_strict_contrib_test if pytype else py_test

    tests = []

    def device_name_flag(name):
        return ["--%s=%s" % (device_flag, name)] if device_flag else []

    if run_on_cpu:
        py_test_macro(
            name = name,
            deps = deps,
            tags = tags + tags_cpu,
            args = args + args_cpu + device_name_flag("cpu"),
            **kwargs
        )
        tests.append(name)

    # NOTE: Running on GPU requires compilation with CUDA (--config=cuda). It's
    # safe to run all test with the cuda config.
    # See: go/forge-for-ml#gpus for available GPU tags.
    for (gpu_name, enable_platform, requires_gpu_tag) in [
        ("gpu", run_on_gpu, "requires-gpu-nvidia"),
        ("gpu_nvidiax2", run_on_gpu_nvidiax2, "requires-gpu-nvidia:2"),
        ("gpu_a100", run_on_gpu_a100, "requires-gpu-sm80"),
        ("gpu_h100", run_on_gpu_h100, "requires-gpu-sm90-full"),
        ("gpu_h100x2", run_on_gpu_h100x2, "requires-gpu-sm90-full:2"),
    ]:
        if enable_platform:
            py_test_macro(
                name = name + "_" + gpu_name,
                deps = deps,
                # NOTE: msan may not support binary blobs like CUDA (c.f. b/33479843).
                # NOTE: tsan does not support binary blobs like CUDA.
                tags = tags + tags_gpu + [requires_gpu_tag, "gpu", "nomsan", "not_run:arm", "notsan"],
                # Don't run GPU-requiring tests on non-x86 platforms, since google3 only has CUDA binary
                # blobs for x86_64. The `not_run:arm` tag is for TAP; target_compatible_with is for
                # manual Blaze invocations.
                target_compatible_with = [
                    "//third_party/bazel_platforms/cpu:x86_64",
                ],
                args = args + args_gpu + device_name_flag(gpu_name),
                **kwargs
            )
            tests.append(name + "_" + gpu_name)

    for (tpu_name, enable_platform, requires_tpu_tag, config_name, chips_per_host_bounds) in [
        # See: go/forge-for-ml#tpus for available TPU tags.
        ("jf", run_on_jf, "requires-jellyfish", None, "1,1,1"),
        ("df", run_on_df, "requires-dragonfish", None, "1,1,1"),
        ("df_2x2", run_on_df_2x2, "requires-dragonfish:4", None, "2,2,1"),
        ("pl", run_on_pl, "requires-puffylite", None, "1,1,1"),
        ("pf", run_on_pf, "requires-pufferfish", "legacy", "1,1,1"),
        ("pf_megacore", run_on_pf_megacore, "requires-pufferfish", "megacore", "1,1,1"),
        ("pf_2x2x1", run_on_pf_2x2x1, "requires-pufferfish:4", "legacy", "2,2,1"),
        ("pf_2x2x1_megacore", run_on_pf_2x2x1_megacore, "requires-pufferfish:4", "megacore", "2,2,1"),
        ("vl", run_on_vl, "requires-viperlite", None, "1,1,1"),
        ("vl_4x2x1", run_on_vl_4x2x1, "requires-viperlite:8", None, "4,2,1"),
        ("vf_megachip", run_on_vf_megachip, "requires-viperfish", "megachip", "1,1,1"),
        ("vf_2x2x1_megachip", run_on_vf_2x2x1_megachip, "requires-viperfish:4", "megachip", "2,2,1"),
        ("glp_1x1", run_on_glp_1x1, "requires-ghostlite", None, "1,1,1"),
        ("glp_2x4", run_on_glp_2x4, "requires-ghostlite:8", None, "2,4,1"),
        ("gf_1x1x1", run_on_gf_1x1x1, "requires-ghostfish", None, "1,1,1"),
        ("gf_2x2x1", run_on_gf_2x2x1, "requires-ghostfish:4", None, "2,2,1"),
    ]:
        if enable_platform:
            is_megacore = config_name in ["megacore", "megachip"]
            deepsea_args = [
                "--deepsea_host_bounds=1,1,1",
                "--deepsea_chips_per_host_bounds=" + chips_per_host_bounds,
            ]
            if config_name != None:
                deepsea_args.append("--deepsea_chip_config_name=" + config_name)

            # --deepsea_* flags must come before any "--" in `args`. `args_tpu` comes after `args`
            # to allow overriding flag values for TPU tests. `args_tpu_megacore` comes after
            # `args_tpu` for the same reason.
            test_args = deepsea_args + args + args_tpu
            if is_megacore:
                test_args = test_args + args_tpu_megacore
            py_test_macro(
                name = name + "_tpu_" + tpu_name,
                deps = deps + ["//learning/brain/google/xla:deepsea_hardware_device"],
                tags = tags + tags_tpu + [requires_tpu_tag, "tpu", "cpu:8", "not_run:arm"],
                # Don't run TPU-requiring tests on non-x86 platforms, since no such hardware
                # exists in Borg. The `not_run:arm` tag is for TAP; target_compatible_with is for
                # manual Blaze invocations.
                target_compatible_with = [
                    "//third_party/bazel_platforms/cpu:x86_64",
                ],
                args = test_args + device_name_flag(tpu_name),
                malloc = "//third_party/tcmalloc",
                **kwargs
            )
            tests.append(name + "_tpu_" + tpu_name)

    # Checks that at least one test target is defined.
    if not tests:
        fail("No test targets generated. Remember to specify at least one platform" +
             "(e.g. run_on_cpu=True).")
    native.test_suite(
        name = name + "_all",
        tags = ["manual"] if "manual" in tags else [],
        tests = tests,
    )

# Register new extensions. go/build-cleaner-build-extensions
register_extension_info(
    extension = py3_binary,
    label_regex_map = {
        "deps": "deps:{extension_name}",
        "pytype_deps": "pytype_deps:{extension_name}",
    },
)

register_extension_info(
    extension = py3_library,
    label_regex_map = {
        "deps": "deps:{extension_name}",
        "pytype_deps": "pytype_deps:{extension_name}",
    },
)

register_extension_info(
    extension = py3_test,
    label_regex_for_dep = "{extension_name}",
)
