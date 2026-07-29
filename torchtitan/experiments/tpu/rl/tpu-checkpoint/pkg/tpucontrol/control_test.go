//go:build linux

package tpucontrol_test

import (
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gpu-os/tpu-checkpoint/pkg/tpucontrol"
)

func setupPipeTest(t *testing.T, pid int, pipes []pipePair) {
	t.Helper()
	root := t.TempDir()
	old := tpucontrol.ProcRoot
	tpucontrol.ProcRoot = root
	t.Cleanup(func() { tpucontrol.ProcRoot = old })

	pidDir := filepath.Join(root, itoa(pid))
	os.MkdirAll(pidDir, 0o755)

	oldOpen := tpucontrol.OpenPipeFile
	t.Cleanup(func() { tpucontrol.OpenPipeFile = oldOpen })

	fdMap := make(map[string]*os.File)
	for i, pp := range pipes {
		tid := 1000 + i
		threadName := pp.threadName
		tidDir := filepath.Join(pidDir, "task", itoa(tid))
		os.MkdirAll(tidDir, 0o755)
		os.WriteFile(filepath.Join(tidDir, "comm"), []byte(threadName+"\n"), 0o644)

		reqPath := tpucontrol.PipePath(pid, pp.reqWriteFD)
		rspPath := tpucontrol.PipePath(pid, pp.rspReadFD)
		fdMap[reqPath] = pp.reqWriteFile
		fdMap[rspPath] = pp.rspReadFile
	}

	tpucontrol.OpenPipeFile = func(path string, flag int) (*os.File, error) {
		if f, ok := fdMap[path]; ok {
			return f, nil
		}
		return nil, os.ErrNotExist
	}
}

type pipePair struct {
	threadName   string
	reqWriteFD   int
	rspReadFD    int
	reqWriteFile *os.File
	rspReadFile  *os.File
}

func marshalResponse(success bool, state tpucontrol.RuntimeState, errMsg string) []byte {
	var buf []byte
	if success {
		buf = append(buf, 0x08, 0x01)
	}
	if state != 0 {
		buf = append(buf, 0x10)
		v := uint64(state)
		for v >= 0x80 {
			buf = append(buf, byte(v)|0x80)
			v >>= 7
		}
		buf = append(buf, byte(v))
	}
	if errMsg != "" {
		buf = append(buf, 0x1a)
		v := uint64(len(errMsg))
		for v >= 0x80 {
			buf = append(buf, byte(v)|0x80)
			v >>= 7
		}
		buf = append(buf, byte(v))
		buf = append(buf, []byte(errMsg)...)
	}
	return buf
}

func writeDelimited(f *os.File, msg []byte) {
	var sizeBuf [4]byte
	binary.BigEndian.PutUint32(sizeBuf[:], uint32(len(msg)))
	f.Write(sizeBuf[:])
	f.Write(msg)
}

func mockResponder(t *testing.T, reqRead, rspWrite *os.File, success bool, state tpucontrol.RuntimeState, errMsg string) {
	t.Helper()
	go func() {
		defer rspWrite.Close()
		// Read the request (discard)
		var sizeBuf [4]byte
		if _, err := reqRead.Read(sizeBuf[:]); err != nil {
			return
		}
		size := binary.BigEndian.Uint32(sizeBuf[:])
		discard := make([]byte, size)
		reqRead.Read(discard)

		// Write response
		resp := marshalResponse(success, state, errMsg)
		writeDelimited(rspWrite, resp)
	}()
}

func TestCheckpoint_Success(t *testing.T) {
	reqRead, reqWrite, _ := os.Pipe()
	rspRead, rspWrite, _ := os.Pipe()
	defer reqRead.Close()

	setupPipeTest(t, 42, []pipePair{{
		threadName:   "libtpu00010002",
		reqWriteFD:   1,
		rspReadFD:    2,
		reqWriteFile: reqWrite,
		rspReadFile:  rspRead,
	}})

	mockResponder(t, reqRead, rspWrite, true, tpucontrol.StateDetached, "")

	if err := tpucontrol.Checkpoint(42, 10); err != nil {
		t.Fatalf("Checkpoint() error = %v", err)
	}
}

func TestRestore_Success(t *testing.T) {
	reqRead, reqWrite, _ := os.Pipe()
	rspRead, rspWrite, _ := os.Pipe()
	defer reqRead.Close()

	setupPipeTest(t, 42, []pipePair{{
		threadName:   "libtpu00010002",
		reqWriteFD:   1,
		rspReadFD:    2,
		reqWriteFile: reqWrite,
		rspReadFile:  rspRead,
	}})

	mockResponder(t, reqRead, rspWrite, true, tpucontrol.StateRunning, "")

	if err := tpucontrol.Restore(42, 10); err != nil {
		t.Fatalf("Restore() error = %v", err)
	}
}

func TestGetState_Success(t *testing.T) {
	reqRead, reqWrite, _ := os.Pipe()
	rspRead, rspWrite, _ := os.Pipe()
	defer reqRead.Close()

	setupPipeTest(t, 42, []pipePair{{
		threadName:   "libtpu00010002",
		reqWriteFD:   1,
		rspReadFD:    2,
		reqWriteFile: reqWrite,
		rspReadFile:  rspRead,
	}})

	mockResponder(t, reqRead, rspWrite, true, tpucontrol.StateDetached, "")

	state, err := tpucontrol.GetState(42, 10)
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if state != tpucontrol.StateDetached {
		t.Errorf("GetState() = %v, want %v", state, tpucontrol.StateDetached)
	}
}

func TestControlOne_FailureResponse(t *testing.T) {
	reqRead, reqWrite, _ := os.Pipe()
	rspRead, rspWrite, _ := os.Pipe()
	defer reqRead.Close()

	setupPipeTest(t, 42, []pipePair{{
		threadName:   "libtpu00010002",
		reqWriteFD:   1,
		rspReadFD:    2,
		reqWriteFile: reqWrite,
		rspReadFile:  rspRead,
	}})

	mockResponder(t, reqRead, rspWrite, false, tpucontrol.StateFaulted, "device busy")

	err := tpucontrol.Checkpoint(42, 10)
	if err == nil {
		t.Fatal("expected error for failure response")
	}

	var cfe *tpucontrol.ControlFailedError
	if !errors.As(err, &cfe) {
		t.Fatalf("expected ControlFailedError, got %T: %v", err, err)
	}
	if cfe.ErrorMessage != "device busy" {
		t.Errorf("ErrorMessage = %q, want %q", cfe.ErrorMessage, "device busy")
	}
	if cfe.CurrentState != tpucontrol.StateFaulted {
		t.Errorf("CurrentState = %v, want %v", cfe.CurrentState, tpucontrol.StateFaulted)
	}
}

func TestControlOne_Timeout(t *testing.T) {
	reqRead, reqWrite, _ := os.Pipe()
	rspRead, rspWrite, _ := os.Pipe()
	defer reqRead.Close()
	defer rspWrite.Close()

	setupPipeTest(t, 42, []pipePair{{
		threadName:   "libtpu00010002",
		reqWriteFD:   1,
		rspReadFD:    2,
		reqWriteFile: reqWrite,
		rspReadFile:  rspRead,
	}})

	// Goroutine reads request but never writes response
	go func() {
		var sizeBuf [4]byte
		reqRead.Read(sizeBuf[:])
		size := binary.BigEndian.Uint32(sizeBuf[:])
		discard := make([]byte, size)
		reqRead.Read(discard)
		time.Sleep(5 * time.Second)
	}()

	err := tpucontrol.Checkpoint(42, 1)
	if err == nil {
		t.Fatal("expected timeout error")
	}

	var te *tpucontrol.TimeoutError
	if !errors.As(err, &te) {
		// Pipe deadlines may produce ProtocolError on some kernels
		var pe *tpucontrol.ProtocolError
		if !errors.As(err, &pe) {
			t.Fatalf("expected TimeoutError or ProtocolError, got %T: %v", err, err)
		}
	}
}

func TestControlOne_PipeClosed(t *testing.T) {
	reqRead, reqWrite, _ := os.Pipe()
	rspRead, rspWrite, _ := os.Pipe()
	defer reqRead.Close()

	setupPipeTest(t, 42, []pipePair{{
		threadName:   "libtpu00010002",
		reqWriteFD:   1,
		rspReadFD:    2,
		reqWriteFile: reqWrite,
		rspReadFile:  rspRead,
	}})

	// Read request then close response pipe without writing
	go func() {
		var sizeBuf [4]byte
		reqRead.Read(sizeBuf[:])
		size := binary.BigEndian.Uint32(sizeBuf[:])
		discard := make([]byte, size)
		reqRead.Read(discard)
		rspWrite.Close()
	}()

	err := tpucontrol.Checkpoint(42, 10)
	if err == nil {
		t.Fatal("expected error for closed pipe")
	}

	var pe *tpucontrol.ProtocolError
	if !errors.As(err, &pe) {
		t.Fatalf("expected ProtocolError, got %T: %v", err, err)
	}
}

func TestControlOne_CorruptedResponse(t *testing.T) {
	reqRead, reqWrite, _ := os.Pipe()
	rspRead, rspWrite, _ := os.Pipe()
	defer reqRead.Close()

	setupPipeTest(t, 42, []pipePair{{
		threadName:   "libtpu00010002",
		reqWriteFD:   1,
		rspReadFD:    2,
		reqWriteFile: reqWrite,
		rspReadFile:  rspRead,
	}})

	go func() {
		defer rspWrite.Close()
		var sizeBuf [4]byte
		reqRead.Read(sizeBuf[:])
		size := binary.BigEndian.Uint32(sizeBuf[:])
		discard := make([]byte, size)
		reqRead.Read(discard)

		// Write garbage: valid size prefix but invalid protobuf (unsupported wire type)
		garbage := []byte{0x0d, 0x01, 0x02, 0x03, 0x04}
		writeDelimited(rspWrite, garbage)
	}()

	err := tpucontrol.Checkpoint(42, 10)
	if err == nil {
		t.Fatal("expected error for corrupted response")
	}

	var pe *tpucontrol.ProtocolError
	if !errors.As(err, &pe) {
		t.Fatalf("expected ProtocolError, got %T: %v", err, err)
	}
}

func TestControlAll_MultipleThreads(t *testing.T) {
	reqRead1, reqWrite1, _ := os.Pipe()
	rspRead1, rspWrite1, _ := os.Pipe()
	reqRead2, reqWrite2, _ := os.Pipe()
	rspRead2, rspWrite2, _ := os.Pipe()
	defer reqRead1.Close()
	defer reqRead2.Close()

	setupPipeTest(t, 42, []pipePair{
		{threadName: "libtpu00010002", reqWriteFD: 1, rspReadFD: 2, reqWriteFile: reqWrite1, rspReadFile: rspRead1},
		{threadName: "libtpu00030004", reqWriteFD: 3, rspReadFD: 4, reqWriteFile: reqWrite2, rspReadFile: rspRead2},
	})

	mockResponder(t, reqRead1, rspWrite1, true, tpucontrol.StateDetached, "")
	mockResponder(t, reqRead2, rspWrite2, true, tpucontrol.StateDetached, "")

	if err := tpucontrol.Checkpoint(42, 10); err != nil {
		t.Fatalf("Checkpoint() error = %v", err)
	}
}

func TestControlAll_FirstFails(t *testing.T) {
	reqRead1, reqWrite1, _ := os.Pipe()
	rspRead1, rspWrite1, _ := os.Pipe()
	reqRead2, reqWrite2, _ := os.Pipe()
	rspRead2, rspWrite2, _ := os.Pipe()
	defer reqRead1.Close()
	defer reqRead2.Close()
	defer rspWrite2.Close()

	setupPipeTest(t, 42, []pipePair{
		{threadName: "libtpu00010002", reqWriteFD: 1, rspReadFD: 2, reqWriteFile: reqWrite1, rspReadFile: rspRead1},
		{threadName: "libtpu00030004", reqWriteFD: 3, rspReadFD: 4, reqWriteFile: reqWrite2, rspReadFile: rspRead2},
	})

	// First thread fails, second should not be contacted
	mockResponder(t, reqRead1, rspWrite1, false, tpucontrol.StateFaulted, "device error")

	err := tpucontrol.Checkpoint(42, 10)
	if err == nil {
		t.Fatal("expected error when first thread fails")
	}

	var cfe *tpucontrol.ControlFailedError
	if !errors.As(err, &cfe) {
		t.Fatalf("expected ControlFailedError, got %T: %v", err, err)
	}
}

func TestCheckProcessExists(t *testing.T) {
	root := t.TempDir()
	old := tpucontrol.ProcRoot
	tpucontrol.ProcRoot = root
	defer func() { tpucontrol.ProcRoot = old }()

	// PID doesn't exist
	err := tpucontrol.CheckProcessExists(12345)
	if err == nil {
		t.Error("expected error for non-existent PID")
	}
	var de *tpucontrol.DiscoveryError
	if !errors.As(err, &de) {
		t.Fatalf("expected DiscoveryError, got %T", err)
	}

	// Create PID dir
	os.MkdirAll(filepath.Join(root, "12345"), 0o755)
	if err := tpucontrol.CheckProcessExists(12345); err != nil {
		t.Errorf("unexpected error for existing PID: %v", err)
	}
}