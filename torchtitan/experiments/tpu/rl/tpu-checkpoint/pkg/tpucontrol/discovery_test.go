package tpucontrol_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/gpu-os/tpu-checkpoint/pkg/tpucontrol"
)

func setupFakeProc(t *testing.T, pid int, threads map[int]string) string {
	t.Helper()
	root := t.TempDir()
	old := tpucontrol.ProcRoot
	tpucontrol.ProcRoot = root
	t.Cleanup(func() { tpucontrol.ProcRoot = old })

	pidDir := filepath.Join(root, itoa(pid))
	os.MkdirAll(pidDir, 0o755)

	for tid, name := range threads {
		tidDir := filepath.Join(pidDir, "task", itoa(tid))
		os.MkdirAll(tidDir, 0o755)
		os.WriteFile(filepath.Join(tidDir, "comm"), []byte(name+"\n"), 0o644)
	}
	return root
}

func itoa(n int) string {
	return filepath.Base(filepath.Join("/", func() string {
		s := ""
		if n == 0 {
			return "0"
		}
		for n > 0 {
			s = string(rune('0'+n%10)) + s
			n /= 10
		}
		return s
	}()))
}

func TestDiscoverPipes_SingleThread(t *testing.T) {
	setupFakeProc(t, 1234, map[int]string{
		100: "libtpu00010002",
	})

	pipes, err := tpucontrol.DiscoverPipes(1234)
	if err != nil {
		t.Fatalf("DiscoverPipes() error = %v", err)
	}
	if len(pipes) != 1 {
		t.Fatalf("got %d pipes, want 1", len(pipes))
	}
	if pipes[0].ReqWriteFD != 1 {
		t.Errorf("ReqWriteFD = %d, want 1", pipes[0].ReqWriteFD)
	}
	if pipes[0].RspReadFD != 2 {
		t.Errorf("RspReadFD = %d, want 2", pipes[0].RspReadFD)
	}
	if pipes[0].TID != 100 {
		t.Errorf("TID = %d, want 100", pipes[0].TID)
	}
	if pipes[0].ThreadName != "libtpu00010002" {
		t.Errorf("ThreadName = %q, want %q", pipes[0].ThreadName, "libtpu00010002")
	}
}

func TestDiscoverPipes_MultipleThreads(t *testing.T) {
	setupFakeProc(t, 5678, map[int]string{
		200: "libtpu00030004",
		201: "libtpu00050006",
		202: "libtpu00070008",
	})

	pipes, err := tpucontrol.DiscoverPipes(5678)
	if err != nil {
		t.Fatalf("DiscoverPipes() error = %v", err)
	}
	if len(pipes) != 3 {
		t.Fatalf("got %d pipes, want 3", len(pipes))
	}
}

func TestDiscoverPipes_LargeHexFDs(t *testing.T) {
	setupFakeProc(t, 1000, map[int]string{
		300: "libtpu035f0360",
	})

	pipes, err := tpucontrol.DiscoverPipes(1000)
	if err != nil {
		t.Fatalf("DiscoverPipes() error = %v", err)
	}
	if len(pipes) != 1 {
		t.Fatalf("got %d pipes, want 1", len(pipes))
	}
	if pipes[0].ReqWriteFD != 0x035f {
		t.Errorf("ReqWriteFD = %d, want %d", pipes[0].ReqWriteFD, 0x035f)
	}
	if pipes[0].RspReadFD != 0x0360 {
		t.Errorf("RspReadFD = %d, want %d", pipes[0].RspReadFD, 0x0360)
	}
}

func TestDiscoverPipes_InvalidHex(t *testing.T) {
	setupFakeProc(t, 1234, map[int]string{
		100: "libtpuZZZZYYYY",
	})

	_, err := tpucontrol.DiscoverPipes(1234)
	if err == nil {
		t.Error("expected error for invalid hex suffix")
	}
}

func TestDiscoverPipes_WrongSuffixLength(t *testing.T) {
	tests := []struct {
		name       string
		threadName string
	}{
		{"too_short", "libtpu123"},
		{"too_long", "libtpu1234567890"},
		{"just_prefix", "libtpu"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			setupFakeProc(t, 1234, map[int]string{
				100: tc.threadName,
			})
			_, err := tpucontrol.DiscoverPipes(1234)
			if err == nil {
				t.Errorf("expected error for thread name %q", tc.threadName)
			}
		})
	}
}

func TestDiscoverPipes_NoLibtpuThreads(t *testing.T) {
	setupFakeProc(t, 1234, map[int]string{
		100: "main",
		101: "worker-1",
		102: "python3",
	})

	_, err := tpucontrol.DiscoverPipes(1234)
	if err == nil {
		t.Error("expected error when no libtpu threads present")
	}
}

func TestDiscoverPipes_MixedThreads(t *testing.T) {
	setupFakeProc(t, 1234, map[int]string{
		100: "main",
		101: "libtpu00010002",
		102: "worker-1",
		103: "libtpu00030004",
		104: "python3",
		105: "libtpuZZZZ0001", // invalid hex, should be skipped
		106: "libtpu123",      // wrong length, should be skipped
	})

	pipes, err := tpucontrol.DiscoverPipes(1234)
	if err != nil {
		t.Fatalf("DiscoverPipes() error = %v", err)
	}
	if len(pipes) != 2 {
		t.Fatalf("got %d pipes, want 2 (valid libtpu threads only)", len(pipes))
	}
}

func TestDiscoverPipes_NonExistentPID(t *testing.T) {
	root := t.TempDir()
	old := tpucontrol.ProcRoot
	tpucontrol.ProcRoot = root
	t.Cleanup(func() { tpucontrol.ProcRoot = old })

	_, err := tpucontrol.DiscoverPipes(99999)
	if err == nil {
		t.Error("expected error for non-existent PID")
	}
}

func TestDiscoverPipes_EmptyCommFile(t *testing.T) {
	setupFakeProc(t, 1234, map[int]string{
		100: "",
	})

	_, err := tpucontrol.DiscoverPipes(1234)
	if err == nil {
		t.Error("expected error for empty comm file")
	}
}

func TestPipePath(t *testing.T) {
	old := tpucontrol.ProcRoot
	tpucontrol.ProcRoot = "/proc"
	defer func() { tpucontrol.ProcRoot = old }()

	got := tpucontrol.PipePath(1234, 5)
	want := "/proc/1234/fd/5"
	if got != want {
		t.Errorf("PipePath(1234, 5) = %q, want %q", got, want)
	}
}