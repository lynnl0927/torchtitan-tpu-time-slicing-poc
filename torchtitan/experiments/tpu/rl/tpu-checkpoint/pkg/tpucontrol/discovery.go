package tpucontrol

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const threadNamePrefix = "libtpu"

// ProcRoot is the base path for /proc. Override in tests to use a temp directory.
var ProcRoot = "/proc"

// PipeInfo holds the pipe file descriptor numbers extracted from a libtpu thread name.
type PipeInfo struct {
	TID        int
	ThreadName string
	ReqWriteFD int
	RspReadFD  int
}

// DiscoverPipes scans /proc/<pid>/task/*/comm for threads named "libtpu{XXXXYYYY}"
// and extracts the pipe FD numbers encoded in the hex suffix.
func DiscoverPipes(pid int) ([]PipeInfo, error) {
	taskDir := fmt.Sprintf("%s/%d/task", ProcRoot, pid)
	entries, err := os.ReadDir(taskDir)
	if err != nil {
		return nil, fmt.Errorf("read task dir %s: %w", taskDir, err)
	}

	var pipes []PipeInfo
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		tid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}

		commPath := filepath.Join(taskDir, e.Name(), "comm")
		data, err := os.ReadFile(commPath)
		if err != nil {
			continue
		}
		name := strings.TrimSpace(string(data))

		if !strings.HasPrefix(name, threadNamePrefix) {
			continue
		}
		hexSuffix := strings.TrimPrefix(name, threadNamePrefix)
		if len(hexSuffix) != 8 {
			continue
		}

		reqFD, err := strconv.ParseInt(hexSuffix[:4], 16, 32)
		if err != nil {
			continue
		}
		rspFD, err := strconv.ParseInt(hexSuffix[4:], 16, 32)
		if err != nil {
			continue
		}

		pipes = append(pipes, PipeInfo{
			TID:        tid,
			ThreadName: name,
			ReqWriteFD: int(reqFD),
			RspReadFD:  int(rspFD),
		})
	}

	if len(pipes) == 0 {
		return nil, fmt.Errorf("no libtpu threads found for PID %d", pid)
	}
	return pipes, nil
}

// PipePath returns the /proc path to access a file descriptor of another process.
func PipePath(pid, fd int) string {
	return fmt.Sprintf("%s/%d/fd/%d", ProcRoot, pid, fd)
}