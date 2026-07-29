// mock-libtpu simulates libtpu's pipe-based control protocol for testing
// tpu-checkpoint locally without real TPU hardware.
//
// It creates anonymous pipes, spawns a thread named "libtpu{XXXXYYYY}" (encoding
// pipe FD numbers in hex), and listens for protobuf ControlRequest messages.
//
// Usage:
//
//	go run ./cmd/mock-libtpu
//	# In another terminal:
//	sudo ./bin/x86_64_Linux/tpu-checkpoint --action checkpoint --pid <mock-pid>
package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"
	"unsafe"

	"github.com/gpu-os/tpu-checkpoint/pkg/tpucontrol"
)

func main() {
	// Create the request pipe (CLI writes, we read)
	reqRead, reqWrite, err := os.Pipe()
	if err != nil {
		fatal("create request pipe: %v", err)
	}

	// Create the response pipe (we write, CLI reads)
	rspRead, rspWrite, err := os.Pipe()
	if err != nil {
		fatal("create response pipe: %v", err)
	}

	// The CLI discovers pipes by thread name. The thread name encodes:
	//   - reqWriteFD: the write end of the request pipe (CLI writes here)
	//   - rspReadFD:  the read end of the response pipe (CLI reads here)
	reqWriteFD := int(reqWrite.Fd())
	rspReadFD := int(rspRead.Fd())

	threadName := fmt.Sprintf("libtpu%04x%04x", reqWriteFD, rspReadFD)

	fmt.Printf("mock-libtpu starting\n")
	fmt.Printf("  PID:           %d\n", os.Getpid())
	fmt.Printf("  req pipe:      read=fd%d  write=fd%d\n", reqRead.Fd(), reqWriteFD)
	fmt.Printf("  rsp pipe:      read=fd%d  write=fd%d\n", rspReadFD, rspWrite.Fd())
	fmt.Printf("  thread name:   %s\n", threadName)
	fmt.Printf("\nTest with:\n")
	fmt.Printf("  sudo tpu-checkpoint --action checkpoint --pid %d\n", os.Getpid())
	fmt.Printf("  sudo tpu-checkpoint --get-state --pid %d\n", os.Getpid())
	fmt.Printf("  sudo tpu-checkpoint --action restore --pid %d\n\n", os.Getpid())

	// Lock this goroutine to an OS thread so we can set the thread name.
	runtime.LockOSThread()
	prSetName(threadName)

	state := tpucontrol.StateRunning
	fmt.Printf("listening for control messages (state=%s)...\n", state)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		fmt.Println("\nshutting down")
		os.Exit(0)
	}()

	for {
		msgData, err := tpucontrol.ReadDelimited(reqRead)
		if err != nil {
			fmt.Printf("error reading request: %v\n", err)
			time.Sleep(100 * time.Millisecond)
			continue
		}

		var req controlRequest
		if err := req.unmarshal(msgData); err != nil {
			fmt.Printf("error unmarshaling request: %v\n", err)
			sendError(rspWrite, "unmarshal error", state)
			continue
		}

		action := tpucontrol.ControlAction(req.action)
		fmt.Printf("received %s (timeout=%ds)\n", action, req.timeoutSecs)

		switch action {
		case tpucontrol.ActionCheckpoint:
			fmt.Println("  simulating checkpoint (500ms)...")
			time.Sleep(500 * time.Millisecond)
			state = tpucontrol.StateDetached
			sendSuccess(rspWrite, state)
			fmt.Printf("  done (state=%s)\n", state)

		case tpucontrol.ActionRestore:
			fmt.Println("  simulating restore (500ms)...")
			time.Sleep(500 * time.Millisecond)
			state = tpucontrol.StateRunning
			sendSuccess(rspWrite, state)
			fmt.Printf("  done (state=%s)\n", state)

		case tpucontrol.ActionGetState:
			sendSuccess(rspWrite, state)
			fmt.Printf("  reported state=%s\n", state)

		default:
			sendError(rspWrite, fmt.Sprintf("unsupported action: %d", req.action), state)
		}
	}
}

type controlRequest struct {
	action      int32
	timeoutSecs int32
}

func (r *controlRequest) unmarshal(data []byte) error {
	for len(data) > 0 {
		if len(data) < 1 {
			break
		}
		tag := data[0]
		fieldNum := tag >> 3
		wireType := tag & 0x7

		if wireType != 0 {
			return fmt.Errorf("unsupported wire type %d", wireType)
		}

		data = data[1:]
		val, n := decodeVarintSimple(data)
		data = data[n:]

		switch fieldNum {
		case 1:
			r.action = int32(val)
		case 2:
			r.timeoutSecs = int32(val)
		}
	}
	return nil
}

func decodeVarintSimple(data []byte) (uint64, int) {
	var val uint64
	for i := 0; i < len(data) && i < 10; i++ {
		b := data[i]
		val |= uint64(b&0x7f) << (7 * i)
		if b < 0x80 {
			return val, i + 1
		}
	}
	return val, 0
}

func sendSuccess(w *os.File, state tpucontrol.RuntimeState) {
	writeDelimited(w, marshalResponse(true, state, ""))
}

func sendError(w *os.File, errMsg string, state tpucontrol.RuntimeState) {
	writeDelimited(w, marshalResponse(false, state, errMsg))
}

func marshalResponse(success bool, state tpucontrol.RuntimeState, errMsg string) []byte {
	var buf []byte
	if success {
		buf = append(buf, 0x08, 0x01) // field 1, varint, value=1
	}
	if state != 0 {
		buf = append(buf, 0x10) // field 2, varint
		buf = appendVarint(buf, uint64(state))
	}
	if errMsg != "" {
		buf = append(buf, 0x1a) // field 3, length-delimited
		buf = appendVarint(buf, uint64(len(errMsg)))
		buf = append(buf, []byte(errMsg)...)
	}
	return buf
}

func appendVarint(buf []byte, val uint64) []byte {
	for val >= 0x80 {
		buf = append(buf, byte(val)|0x80)
		val >>= 7
	}
	return append(buf, byte(val))
}

func writeDelimited(w *os.File, msg []byte) {
	var sizeBuf [4]byte
	binary.BigEndian.PutUint32(sizeBuf[:], uint32(len(msg)))
	w.Write(sizeBuf[:])
	w.Write(msg)
}

// prSetName sets the calling thread's comm name via prctl(PR_SET_NAME).
func prSetName(name string) {
	buf := make([]byte, 16)
	copy(buf, name)
	syscall.Syscall6(syscall.SYS_PRCTL, 15, uintptr(unsafe.Pointer(&buf[0])), 0, 0, 0, 0)
}

func fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "mock-libtpu: "+format+"\n", args...)
	os.Exit(1)
}