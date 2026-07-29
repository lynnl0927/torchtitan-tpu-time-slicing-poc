package tpucontrol

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// ControlAction matches gVisor's tpu_control.proto ControlAction enum.
type ControlAction int32

const (
	ActionUnspecified ControlAction = 0
	ActionLock        ControlAction = 1
	ActionCheckpoint  ControlAction = 2
	ActionRestore     ControlAction = 3
	ActionUnlock      ControlAction = 4
	ActionGetState    ControlAction = 5
)

func (a ControlAction) String() string {
	switch a {
	case ActionCheckpoint:
		return "checkpoint"
	case ActionRestore:
		return "restore"
	case ActionGetState:
		return "getstate"
	default:
		return fmt.Sprintf("action(%d)", int32(a))
	}
}

// RuntimeState matches gVisor's tpu_control.proto RuntimeState enum.
type RuntimeState int32

const (
	StateUnspecified RuntimeState = 0
	StateRunning     RuntimeState = 1
	StateLocked      RuntimeState = 2
	StateDetached    RuntimeState = 3
	StateRestoring   RuntimeState = 4
	StateFaulted     RuntimeState = 5
)

func (s RuntimeState) String() string {
	switch s {
	case StateRunning:
		return "running"
	case StateLocked:
		return "locked"
	case StateDetached:
		return "detached"
	case StateRestoring:
		return "restoring"
	case StateFaulted:
		return "faulted"
	default:
		return fmt.Sprintf("state(%d)", int32(s))
	}
}

// ControlRequest is the protobuf message sent to libtpu via the request pipe.
//
//	message ControlRequest {
//	  ControlAction action = 1;
//	  int32 timeout_secs = 2;
//	}
type ControlRequest struct {
	Action      ControlAction
	TimeoutSecs int32
}

// Marshal encodes a ControlRequest to protobuf wire format.
func (r *ControlRequest) Marshal() []byte {
	var buf []byte
	if r.Action != 0 {
		buf = appendVarintField(buf, 1, uint64(r.Action))
	}
	if r.TimeoutSecs != 0 {
		buf = appendVarintField(buf, 2, uint64(r.TimeoutSecs))
	}
	return buf
}

// ControlResponse is the protobuf message read from libtpu via the response pipe.
//
//	message ControlResponse {
//	  bool success = 1;
//	  RuntimeState current_state = 2;
//	  string error_message = 3;
//	}
type ControlResponse struct {
	Success      bool
	CurrentState RuntimeState
	ErrorMessage string
}

// Unmarshal decodes a ControlResponse from protobuf wire format.
func (r *ControlResponse) Unmarshal(data []byte) error {
	for len(data) > 0 {
		fieldNum, wireType, n, err := decodeTag(data)
		if err != nil {
			return fmt.Errorf("decode tag: %w", err)
		}
		data = data[n:]

		switch wireType {
		case 0: // varint
			val, m, err := decodeVarint(data)
			if err != nil {
				return fmt.Errorf("decode varint field %d: %w", fieldNum, err)
			}
			data = data[m:]
			switch fieldNum {
			case 1:
				r.Success = val != 0
			case 2:
				r.CurrentState = RuntimeState(val)
			}
		case 2: // length-delimited
			if len(data) < 1 {
				return fmt.Errorf("truncated length for field %d", fieldNum)
			}
			length, m, err := decodeVarint(data)
			if err != nil {
				return fmt.Errorf("decode length field %d: %w", fieldNum, err)
			}
			data = data[m:]
			if uint64(len(data)) < length {
				return fmt.Errorf("truncated data for field %d: need %d, have %d", fieldNum, length, len(data))
			}
			switch fieldNum {
			case 3:
				r.ErrorMessage = string(data[:length])
			}
			data = data[length:]
		default:
			return fmt.Errorf("unsupported wire type %d for field %d", wireType, fieldNum)
		}
	}
	return nil
}

// WriteDelimited writes a length-delimited protobuf message (4-byte big-endian size prefix).
func WriteDelimited(w io.Writer, msg []byte) error {
	var sizeBuf [4]byte
	binary.BigEndian.PutUint32(sizeBuf[:], uint32(len(msg)))
	if _, err := w.Write(sizeBuf[:]); err != nil {
		return fmt.Errorf("write size prefix: %w", err)
	}
	if _, err := w.Write(msg); err != nil {
		return fmt.Errorf("write message: %w", err)
	}
	return nil
}

// ReadDelimited reads a length-delimited protobuf message (4-byte big-endian size prefix).
func ReadDelimited(r io.Reader) ([]byte, error) {
	var sizeBuf [4]byte
	if _, err := io.ReadFull(r, sizeBuf[:]); err != nil {
		return nil, fmt.Errorf("read size prefix: %w", err)
	}
	size := binary.BigEndian.Uint32(sizeBuf[:])
	if size > 1<<20 {
		return nil, fmt.Errorf("message too large: %d bytes", size)
	}
	msg := make([]byte, size)
	if _, err := io.ReadFull(r, msg); err != nil {
		return nil, fmt.Errorf("read message body (%d bytes): %w", size, err)
	}
	return msg, nil
}

func appendVarintField(buf []byte, fieldNum uint32, val uint64) []byte {
	tag := (fieldNum << 3) | 0 // wire type 0 = varint
	buf = appendVarint(buf, uint64(tag))
	buf = appendVarint(buf, val)
	return buf
}

func appendVarint(buf []byte, val uint64) []byte {
	for val >= 0x80 {
		buf = append(buf, byte(val)|0x80)
		val >>= 7
	}
	buf = append(buf, byte(val))
	return buf
}

func decodeTag(data []byte) (fieldNum uint32, wireType uint8, n int, err error) {
	val, n, err := decodeVarint(data)
	if err != nil {
		return 0, 0, 0, err
	}
	if val > math.MaxUint32 {
		return 0, 0, 0, fmt.Errorf("tag overflow: %d", val)
	}
	return uint32(val >> 3), uint8(val & 0x7), n, nil
}

func decodeVarint(data []byte) (uint64, int, error) {
	var val uint64
	for i := 0; i < len(data) && i < 10; i++ {
		b := data[i]
		val |= uint64(b&0x7f) << (7 * i)
		if b < 0x80 {
			return val, i + 1, nil
		}
	}
	return 0, 0, fmt.Errorf("varint not terminated")
}