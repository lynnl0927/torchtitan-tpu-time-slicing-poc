package tpucontrol

import (
	"errors"
	"fmt"
	"os"
	"time"
)

const defaultTimeoutSecs = 180

// OpenPipeFile opens a pipe by path. Override in tests to supply pre-opened pipes.
var OpenPipeFile = func(path string, flag int) (*os.File, error) {
	return os.OpenFile(path, flag, 0)
}

// CheckProcessExists verifies the target process is alive.
func CheckProcessExists(pid int) error {
	path := fmt.Sprintf("%s/%d", ProcRoot, pid)
	if _, err := os.Stat(path); err != nil {
		return &DiscoveryError{PID: pid, Err: fmt.Errorf("process not found: %w", err)}
	}
	return nil
}

// Checkpoint sends ACTION_CHECKPOINT to all libtpu threads for the given PID.
func Checkpoint(pid int, timeoutSecs int) error {
	return controlAll(pid, ActionCheckpoint, timeoutSecs)
}

// Restore sends ACTION_RESTORE to all libtpu threads for the given PID.
func Restore(pid int, timeoutSecs int) error {
	return controlAll(pid, ActionRestore, timeoutSecs)
}

// GetState sends ACTION_GETSTATE to the first libtpu thread for the given PID
// and returns the current RuntimeState.
func GetState(pid int, timeoutSecs int) (RuntimeState, error) {
	if err := CheckProcessExists(pid); err != nil {
		return StateUnspecified, err
	}
	pipes, err := DiscoverPipes(pid)
	if err != nil {
		return StateUnspecified, &DiscoveryError{PID: pid, Err: err}
	}
	resp, err := controlOne(pid, pipes[0], ActionGetState, timeoutSecs)
	if err != nil {
		return StateUnspecified, err
	}
	return resp.CurrentState, nil
}

func controlAll(pid int, action ControlAction, timeoutSecs int) error {
	if err := CheckProcessExists(pid); err != nil {
		return err
	}

	pipes, err := DiscoverPipes(pid)
	if err != nil {
		return &DiscoveryError{PID: pid, Err: err}
	}

	if timeoutSecs <= 0 {
		timeoutSecs = defaultTimeoutSecs
	}

	for _, p := range pipes {
		if _, err := controlOne(pid, p, action, timeoutSecs); err != nil {
			return err
		}
	}
	return nil
}

func controlOne(pid int, pipe PipeInfo, action ControlAction, timeoutSecs int) (*ControlResponse, error) {
	reqPath := PipePath(pid, pipe.ReqWriteFD)
	rspPath := PipePath(pid, pipe.RspReadFD)

	reqFile, err := OpenPipeFile(reqPath, os.O_WRONLY)
	if err != nil {
		return nil, &ProtocolError{Action: action, TID: pipe.TID, Err: fmt.Errorf("open request pipe %s: %w", reqPath, err)}
	}
	defer reqFile.Close()

	rspFile, err := OpenPipeFile(rspPath, os.O_RDONLY)
	if err != nil {
		return nil, &ProtocolError{Action: action, TID: pipe.TID, Err: fmt.Errorf("open response pipe %s: %w", rspPath, err)}
	}
	defer rspFile.Close()

	if timeoutSecs > 0 {
		if err := rspFile.SetReadDeadline(time.Now().Add(time.Duration(timeoutSecs) * time.Second)); err != nil {
			// SetReadDeadline not supported on this platform/fd type — proceed without timeout
		}
	}

	req := &ControlRequest{
		Action:      action,
		TimeoutSecs: int32(timeoutSecs),
	}
	if err := WriteDelimited(reqFile, req.Marshal()); err != nil {
		return nil, &ProtocolError{Action: action, TID: pipe.TID, Err: fmt.Errorf("write request: %w", err)}
	}

	respData, err := ReadDelimited(rspFile)
	if err != nil {
		if errors.Is(err, os.ErrDeadlineExceeded) {
			return nil, &TimeoutError{Action: action, TID: pipe.TID, Err: err}
		}
		return nil, &ProtocolError{Action: action, TID: pipe.TID, Err: fmt.Errorf("read response: %w", err)}
	}

	var resp ControlResponse
	if err := resp.Unmarshal(respData); err != nil {
		return nil, &ProtocolError{Action: action, TID: pipe.TID, Err: fmt.Errorf("unmarshal response: %w", err)}
	}

	if !resp.Success {
		return nil, &ControlFailedError{
			Action:       action,
			TID:          pipe.TID,
			CurrentState: resp.CurrentState,
			ErrorMessage: resp.ErrorMessage,
		}
	}

	return &resp, nil
}