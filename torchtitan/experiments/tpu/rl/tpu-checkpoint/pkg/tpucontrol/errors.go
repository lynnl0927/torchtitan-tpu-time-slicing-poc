package tpucontrol

import "fmt"

// DiscoveryError indicates failure to find libtpu control threads.
type DiscoveryError struct {
	PID int
	Err error
}

func (e *DiscoveryError) Error() string {
	return fmt.Sprintf("discovery for PID %d: %v", e.PID, e.Err)
}

func (e *DiscoveryError) Unwrap() error { return e.Err }

// TimeoutError indicates the operation timed out waiting for a response.
type TimeoutError struct {
	Action ControlAction
	TID    int
	Err    error
}

func (e *TimeoutError) Error() string {
	return fmt.Sprintf("%s timed out on thread %d: %v", e.Action, e.TID, e.Err)
}

func (e *TimeoutError) Unwrap() error { return e.Err }

// ProtocolError indicates a wire-format or response parsing failure.
type ProtocolError struct {
	Action ControlAction
	TID    int
	Err    error
}

func (e *ProtocolError) Error() string {
	return fmt.Sprintf("%s protocol error on thread %d: %v", e.Action, e.TID, e.Err)
}

func (e *ProtocolError) Unwrap() error { return e.Err }

// ControlFailedError indicates libtpu returned Success=false.
type ControlFailedError struct {
	Action       ControlAction
	TID          int
	CurrentState RuntimeState
	ErrorMessage string
}

func (e *ControlFailedError) Error() string {
	if e.ErrorMessage != "" {
		return fmt.Sprintf("%s failed on thread %d: %s", e.Action, e.TID, e.ErrorMessage)
	}
	return fmt.Sprintf("%s failed on thread %d: state=%s", e.Action, e.TID, e.CurrentState)
}