# Deploying Disaggregated TPU Time-Slicing (GKE)

This guide walks you through deploying the fully disaggregated two-job time-slicing RL workload on Google Kubernetes Engine (GKE) using TPU v5e (`v5litepod-8`).

## 1. Download the Custom `_libtpu.so`

To enable time-slicing (UDS-based checkpoint/restore), you need the custom-built `_libtpu.so` library.

1. Download the `.so` file from the GCS bucket directly into the **root** of your `torchtitan` repository:
   ```bash
   cd /path/to/torchtitan
   gcloud storage cp gs://linglinll-gke-dev-libtpu/_libtpu.so ./_libtpu.so
   ```

Your directory structure should look like this:
```text
torchtitan/
├── _libtpu.so              <-- Place it here!
├── torchtitan/
│   ├── experiments/tpu/rl/rl-disagg/
│   │   ├── deploy/
│   │   │   ├── Dockerfile
│   │   │   └── rl-disagg.yaml
...
```

*(Note: The Dockerfile uses `COPY _libtpu.so /app/_libtpu.so`, which expects the file to be present in the Docker build context—i.e., the repository root).*

## 2. Build and Push the Docker Image

The Dockerfile installs `torch_tpu` from an authenticated artifact registry, so you must pass your GCP token during the build.

1. Set your GCP environment variables:
   ```bash
   export PROJECT_ID="your-gcp-project-id"
   export REGION="us-central1"
   export REPO_NAME="your-artifact-repo"
   export IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/torchtitan-rl-disagg:latest"
   ```

2. Authenticate and obtain an access token:
   ```bash
   gcloud auth login
   gcloud auth configure-docker ${REGION}-docker.pkg.dev
   export GCP_TOKEN=$(gcloud auth print-access-token)
   ```

3. Build the Docker image from the **root of the repository**:
   ```bash
   # Navigate to the root of the torchtitan repository
   cd /path/to/torchtitan
   
   docker build \
     --build-arg GCP_TOKEN=$GCP_TOKEN \
     -t $IMAGE_NAME \
     -f torchtitan/experiments/tpu/rl/rl-disagg/deploy/Dockerfile .
   ```

4. Push the image to Artifact Registry:
   ```bash
   docker push $IMAGE_NAME
   ```

## 3. Running a One-Job Baseline (Without C/R)

If you wish to test the architecture as a standard, single-job disaggregated pipeline without TPU time-slicing (no checkpoint/restore), you can do so by deploying only one job and disabling snapshot mode:

1. **Disable Snapshot Mode**: In the `rl-disagg.yaml` file, locate the `orchestrator-deployment` and change the `MODE` environment variable from `snapshot` to `baseline`.
2. **Remove Job B**: Delete all Kubernetes resources associated with Job B from the manifest (i.e., `trainer-b`, `sampler-b`, `driver-b`, and their corresponding `Service` entries).
3. **Revert to Standard TPU Scheduling (Optional)**: Since Job A is no longer sharing the TPU, you don't need privileged access or custom `libtpu` bypassing:
   - Remove `securityContext: privileged: true` from `sampler-a`.
   - Remove the `hostPath` volumes and `volumeMounts` for `/dev/vfio` and `/var/run/tpu-plugin`.
   - Add back `google.com/tpu: 8` to the `resources: limits:` block for `sampler-a`.
   - Remove the `TPU_LIBRARY_PATH` environment variable so the standard `libtpu.so` is used.
4. **Deploy**:
   ```bash
   kubectl apply -f torchtitan/experiments/tpu/rl/rl-disagg/deploy/rl-disagg.yaml
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
