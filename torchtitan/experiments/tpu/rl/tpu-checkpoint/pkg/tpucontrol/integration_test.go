//go:build linux

package tpucontrol_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gpu-os/tpu-checkpoint/pkg/tpucontrol"
)

func TestIntegration_MockLibtpu(t *testing.T) {
	if os.Getenv("TPU_INTEGRATION") == "" {
		t.Skip("set TPU_INTEGRATION=1 to run integration tests")
	}

	// Ensure ProcRoot points at real /proc for this test
	old := tpucontrol.ProcRoot
	tpucontrol.ProcRoot = "/proc"
	t.Cleanup(func() { tpucontrol.ProcRoot = old })

	oldOpen := tpucontrol.OpenPipeFile
	tpucontrol.OpenPipeFile = func(path string, flag int) (*os.File, error) {
		return os.OpenFile(path, flag, 0)
	}
	t.Cleanup(func() { tpucontrol.OpenPipeFile = oldOpen })

	// Build mock-libtpu
	tmpBin := filepath.Join(t.TempDir(), "mock-libtpu")
	build := exec.Command("go", "build", "-o", tmpBin, "./cmd/mock-libtpu/")
	build.Dir = findModuleRoot(t)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("failed to build mock-libtpu: %v\n%s", err, out)
	}

	// Start mock-libtpu
	cmd := exec.Command(tmpBin)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start mock-libtpu: %v", err)
	}
	pid := cmd.Process.Pid
	t.Cleanup(func() {
		cmd.Process.Kill()
		cmd.Wait()
	})

	// Wait for libtpu thread to appear
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		pipes, err := tpucontrol.DiscoverPipes(pid)
		if err == nil && len(pipes) > 0 {
			t.Logf("discovered pipe: %+v", pipes[0])
			break
		}
		time.Sleep(200 * time.Millisecond)
	}

	pipes, err := tpucontrol.DiscoverPipes(pid)
	if err != nil {
		t.Fatalf("failed to discover pipes for mock PID %d: %v", pid, err)
	}
	if len(pipes) == 0 {
		t.Fatal("no libtpu pipes found")
	}

	t.Run("Checkpoint", func(t *testing.T) {
		if err := tpucontrol.Checkpoint(pid, 10); err != nil {
			t.Fatalf("Checkpoint() error = %v", err)
		}
	})

	t.Run("Restore", func(t *testing.T) {
		if err := tpucontrol.Restore(pid, 10); err != nil {
			t.Fatalf("Restore() error = %v", err)
		}
	})

	t.Run("CheckpointRestoreRoundtrip", func(t *testing.T) {
		if err := tpucontrol.Checkpoint(pid, 10); err != nil {
			t.Fatalf("Checkpoint() error = %v", err)
		}
		if err := tpucontrol.Restore(pid, 10); err != nil {
			t.Fatalf("Restore() error = %v", err)
		}
	})
}

func findModuleRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find go.mod")
		}
		dir = parent
	}
}

func init() {
	// Ensure itoa is available (defined in discovery_test.go)
	_ = strings.TrimSpace
}