package tpucontrol_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/gpu-os/tpu-checkpoint/pkg/tpucontrol"
)

func TestControlRequestMarshal(t *testing.T) {
	tests := []struct {
		name string
		req  tpucontrol.ControlRequest
	}{
		{"zero_value", tpucontrol.ControlRequest{}},
		{"checkpoint_only", tpucontrol.ControlRequest{Action: tpucontrol.ActionCheckpoint}},
		{"restore_only", tpucontrol.ControlRequest{Action: tpucontrol.ActionRestore}},
		{"timeout_only", tpucontrol.ControlRequest{TimeoutSecs: 180}},
		{"both_fields", tpucontrol.ControlRequest{Action: tpucontrol.ActionCheckpoint, TimeoutSecs: 60}},
		{"getstate", tpucontrol.ControlRequest{Action: tpucontrol.ActionGetState, TimeoutSecs: 10}},
		{"large_timeout", tpucontrol.ControlRequest{Action: tpucontrol.ActionRestore, TimeoutSecs: 32767}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data := tc.req.Marshal()
			if data == nil && tc.req.Action == 0 && tc.req.TimeoutSecs == 0 {
				return
			}
			if len(data) == 0 && (tc.req.Action != 0 || tc.req.TimeoutSecs != 0) {
				t.Errorf("expected non-empty marshal output for %+v", tc.req)
			}
		})
	}
}

func TestControlResponseUnmarshal(t *testing.T) {
	tests := []struct {
		name    string
		data    []byte
		want    tpucontrol.ControlResponse
		wantErr bool
	}{
		{
			name: "empty",
			data: []byte{},
			want: tpucontrol.ControlResponse{},
		},
		{
			name: "success_true",
			data: []byte{0x08, 0x01},
			want: tpucontrol.ControlResponse{Success: true},
		},
		{
			name: "state_running",
			data: []byte{0x10, 0x01},
			want: tpucontrol.ControlResponse{CurrentState: tpucontrol.StateRunning},
		},
		{
			name: "state_detached",
			data: []byte{0x10, 0x03},
			want: tpucontrol.ControlResponse{CurrentState: tpucontrol.StateDetached},
		},
		{
			name: "error_message",
			data: append([]byte{0x1a, 0x05}, []byte("error")...),
			want: tpucontrol.ControlResponse{ErrorMessage: "error"},
		},
		{
			name: "all_fields",
			data: append([]byte{0x08, 0x01, 0x10, 0x03, 0x1a, 0x02}, []byte("ok")...),
			want: tpucontrol.ControlResponse{Success: true, CurrentState: tpucontrol.StateDetached, ErrorMessage: "ok"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got tpucontrol.ControlResponse
			err := got.Unmarshal(tc.data)
			if (err != nil) != tc.wantErr {
				t.Fatalf("Unmarshal() error = %v, wantErr %v", err, tc.wantErr)
			}
			if err != nil {
				return
			}
			if got.Success != tc.want.Success {
				t.Errorf("Success = %v, want %v", got.Success, tc.want.Success)
			}
			if got.CurrentState != tc.want.CurrentState {
				t.Errorf("CurrentState = %v, want %v", got.CurrentState, tc.want.CurrentState)
			}
			if got.ErrorMessage != tc.want.ErrorMessage {
				t.Errorf("ErrorMessage = %q, want %q", got.ErrorMessage, tc.want.ErrorMessage)
			}
		})
	}
}

func TestControlResponseUnmarshalErrors(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"truncated_varint", []byte{0x08, 0x80}},
		{"truncated_length", []byte{0x1a, 0x0a, 0x01}},
		{"unsupported_wire_type_1", []byte{0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
		{"unsupported_wire_type_5", []byte{0x0d, 0x00, 0x00, 0x00, 0x00}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var resp tpucontrol.ControlResponse
			if err := resp.Unmarshal(tc.data); err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestWriteDelimitedReadDelimited(t *testing.T) {
	tests := []struct {
		name string
		msg  []byte
	}{
		{"empty", []byte{}},
		{"small", []byte{0x08, 0x02, 0x10, 0xb4, 0x01}},
		{"medium", bytes.Repeat([]byte{0xab}, 256)},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := tpucontrol.WriteDelimited(&buf, tc.msg); err != nil {
				t.Fatalf("WriteDelimited() error = %v", err)
			}
			got, err := tpucontrol.ReadDelimited(&buf)
			if err != nil {
				t.Fatalf("ReadDelimited() error = %v", err)
			}
			if !bytes.Equal(got, tc.msg) {
				t.Errorf("round-trip mismatch: got %v, want %v", got, tc.msg)
			}
		})
	}
}

func TestWriteDelimitedMultiple(t *testing.T) {
	var buf bytes.Buffer
	msgs := [][]byte{{0x01}, {0x02, 0x03}, {0x04, 0x05, 0x06}}
	for _, msg := range msgs {
		if err := tpucontrol.WriteDelimited(&buf, msg); err != nil {
			t.Fatalf("WriteDelimited() error = %v", err)
		}
	}
	for i, want := range msgs {
		got, err := tpucontrol.ReadDelimited(&buf)
		if err != nil {
			t.Fatalf("ReadDelimited()[%d] error = %v", i, err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("[%d] got %v, want %v", i, got, want)
		}
	}
}

func TestReadDelimitedErrors(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"empty", []byte{}},
		{"truncated_size", []byte{0x00, 0x00}},
		{"truncated_body", []byte{0x00, 0x00, 0x00, 0x0a, 0x01, 0x02}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := bytes.NewReader(tc.data)
			if _, err := tpucontrol.ReadDelimited(r); err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestReadDelimitedOversized(t *testing.T) {
	var buf bytes.Buffer
	// Write a size prefix claiming 2MB (exceeds 1<<20 limit)
	buf.Write([]byte{0x00, 0x20, 0x00, 0x00})
	if _, err := tpucontrol.ReadDelimited(&buf); err == nil {
		t.Error("expected error for oversized message")
	}
}

func TestVarintRoundTrip(t *testing.T) {
	values := []uint64{0, 1, 127, 128, 255, 256, 16383, 16384, math.MaxUint32, math.MaxUint64}
	for _, v := range values {
		buf := tpucontrol.ExportAppendVarint(nil, v)
		got, n, err := tpucontrol.ExportDecodeVarint(buf)
		if err != nil {
			t.Errorf("decodeVarint(%d) error = %v", v, err)
			continue
		}
		if n != len(buf) {
			t.Errorf("decodeVarint(%d) consumed %d bytes, encoded %d", v, n, len(buf))
		}
		if got != v {
			t.Errorf("decodeVarint round-trip: got %d, want %d", got, v)
		}
	}
}

func TestControlActionString(t *testing.T) {
	tests := []struct {
		action tpucontrol.ControlAction
		want   string
	}{
		{tpucontrol.ActionCheckpoint, "checkpoint"},
		{tpucontrol.ActionRestore, "restore"},
		{tpucontrol.ActionGetState, "getstate"},
		{tpucontrol.ControlAction(99), "action(99)"},
	}
	for _, tc := range tests {
		if got := tc.action.String(); got != tc.want {
			t.Errorf("ControlAction(%d).String() = %q, want %q", int(tc.action), got, tc.want)
		}
	}
}

func TestRuntimeStateString(t *testing.T) {
	tests := []struct {
		state tpucontrol.RuntimeState
		want  string
	}{
		{tpucontrol.StateRunning, "running"},
		{tpucontrol.StateDetached, "detached"},
		{tpucontrol.StateFaulted, "faulted"},
		{tpucontrol.RuntimeState(99), "state(99)"},
	}
	for _, tc := range tests {
		if got := tc.state.String(); got != tc.want {
			t.Errorf("RuntimeState(%d).String() = %q, want %q", int(tc.state), got, tc.want)
		}
	}
}