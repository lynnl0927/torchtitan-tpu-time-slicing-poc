# Deploying Disaggregated TPU Time-Slicing (GKE)

This guide walks you through deploying the fully disaggregated two-job time-slicing RL workload on Google Kubernetes Engine (GKE) using TPU v5e (`v5litepod-8`).

## 1. The Custom libtpu (`_libtpuv2.so`)

Time-slicing requires a custom libtpu build with checkpoint/restore support.
The current build is **`gs://linglinll-gke-dev-libtpu/_libtpuv2.so`** (pipe-based
`tpu_control` C/R + working duty-cycle metrics; see the main
[README §4](../README.md)). You do **not** need to download it or bake it into
the image: `rl-disagg.yaml` stages it into each TPU worker with a
`stage-libtpu` initContainer and selects it via
`TPU_LIBRARY_PATH=/shared/_libtpuv2.so`, alongside
`LIBTPU_CHECKPOINTING_ENABLED=true` (both are required for C/R to exist at all
in v2). To roll out a new libtpu build, upload it to the bucket and update the
two references in the manifest — no image rebuild needed.

The GCS bucket is the **only** source of the checkpointing libtpu — the image
does not bake one in. Without the manifest's `TPU_LIBRARY_PATH` override, the
pip-installed `libtpu` package is used (fine for plain runs; no C/R).

> [!IMPORTANT]
> The image must contain the **v2-aware** `tpu_hal_snapshot.py` (pipe protocol)
> and `orchestrator.py` from this directory. If your image tag predates those
> changes, rebuild it (step 2) — or overlay the two files with a ConfigMap
> mounted via `subPath` over
> `/app/torchtitan/torchtitan/experiments/tpu/rl/rl-disagg/`.

## 2. Build and Push the Docker Image

The internal `torch_tpu` wheel is installed from a local `wheels/` directory in
the build context (a token build-arg would leak the credential into the image's
build history, so we pre-download instead).

1. Set your GCP environment variables:
   ```bash
   export PROJECT_ID="your-gcp-project-id"
   export REGION="us-central1"
   export REPO_NAME="your-artifact-repo"
   export IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/torchtitan-rl:libtpuv2"
   ```

2. Pre-download the `torch_tpu` wheel into the repo root (gitignored). The
   wheels are linux/x86_64 (`manylinux_2_31`, cp312) — pass the platform flags
   when downloading from a non-linux machine:
   ```bash
   cd /path/to/torchtitan && mkdir -p wheels
   pip download --pre --no-deps --only-binary=:all: \
     --platform manylinux_2_31_x86_64 --python-version 3.12 \
     --implementation cp --abi cp312 \
     --index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
     torch_tpu -d wheels/
   ```

3. Build and push from the **root of the repository** — either with Cloud
   Build (no local Docker needed; keep a `.gcloudignore` that excludes `.git/`
   but NOT `wheels/`):
   ```bash
   gcloud builds submit --project $PROJECT_ID --region $REGION \
     --config torchtitan/experiments/tpu/rl/rl-disagg/deploy/cloudbuild.yaml .
   ```
   or locally (the `RUN --mount` in the Dockerfile requires BuildKit):
   ```bash
   gcloud auth configure-docker ${REGION}-docker.pkg.dev
   DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
     -t $IMAGE_NAME \
     -f torchtitan/experiments/tpu/rl/rl-disagg/deploy/Dockerfile .
   docker push $IMAGE_NAME
   ```

Note the image no longer contains any libtpu at `/app/_libtpu.so` — the
checkpointing libtpu comes solely from the GCS bucket at pod startup (§1).

## 3. Running a One-Job Baseline (Without C/R)

If you wish to test the architecture as a standard, single-job disaggregated pipeline without TPU time-slicing (no checkpoint/restore), you can use the provided baseline manifest. The baseline drops the second job and ensures the primary job correctly claims native TPU resources without sharing.

1. **Update Image URI**: 
   Update the `image:` fields to match your newly pushed `$IMAGE_NAME`:
   ```bash
   sed -i "s|image: .*|image: $IMAGE_NAME|g" torchtitan/experiments/tpu/rl/rl-disagg/deploy/rl-disagg-baseline.yaml
   ```

2. **Deploy**:
   ```bash
   kubectl apply -f torchtitan/experiments/tpu/rl/rl-disagg/deploy/rl-disagg-baseline.yaml
   ```

## 4. Deploying Two-Job Time Slicing (With C/R)

1. **Update Image URI**: 
   Open `torchtitan/experiments/tpu/rl/rl-disagg/deploy/rl-disagg.yaml` and update the `image:` fields across the manifest to match your newly pushed `$IMAGE_NAME`.
   
   Alternatively, you can run this command to update it automatically (replace the placeholder if needed):
   ```bash
   sed -i "s|image: .*|image: $IMAGE_NAME|g" torchtitan/experiments/tpu/rl/rl-disagg/deploy/rl-disagg.yaml
   ```

2. **Connect to your GKE Cluster**:
   ```bash
   gcloud container clusters get-credentials your-cluster-name \
       --region $REGION \
       --project $PROJECT_ID
   ```

3. **Deploy the Resources**:
   Apply the Kubernetes manifest:
   ```bash
   kubectl apply -f torchtitan/experiments/tpu/rl/rl-disagg/deploy/rl-disagg.yaml
   ```

## 5. Verify and Monitor

The deployment creates an orchestrator service along with separated pods for jobs `A` and `B` (`trainer-a`, `sampler-a`, `driver-a`, etc.). Time-slicing is orchestrated automatically through the orchestrator lock.

1. Check that all pods are running and assigned correctly:
   ```bash
   kubectl get pods
   ```

2. Monitor the driver logs to observe generation, training, and the orchestrator acquiring locks:
   ```bash
   # Follow Job A's progress
   kubectl logs -l app=driver-a -f
   
   # Or follow Job B's progress
   kubectl logs -l app=driver-b -f
   ```

3. Verify time-slicing is working correctly by checking the `sampler` logs. You should see `[TPU Snapshot]` checkpoint/restore sequences indicating that the TPUs are successfully being yielded back and forth between Pods:
   ```bash
   kubectl logs -l app=sampler-a -f
   ```
