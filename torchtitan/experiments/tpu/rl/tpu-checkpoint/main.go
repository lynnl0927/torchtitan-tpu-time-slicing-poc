package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gpu-os/tpu-checkpoint/pkg/tpucontrol"
)

const version = "0.2.0"

func usage() {
	fmt.Fprintf(os.Stderr, `tpu-checkpoint: checkpoint and restore TPU processes.
Version %s.

Discovers libtpu control pipes by scanning thread names in /proc/<pid>/task/,
then sends protobuf control messages matching gVisor's tpu_control protocol.

Operations:
  --action checkpoint | restore   --pid <pid> [--timeout <seconds>]
        Checkpoint or restore the TPU runtime for <pid>.

  --get-state --pid <pid>
        Prints the current TPU runtime state.

Options:
  --pid|-p <pid>           Target process PID
  --timeout|-t <seconds>   Operation timeout in seconds (default: 180)
  --help|-h                Print this help message

Discovery:
  Scans /proc/<pid>/task/*/comm for threads named "libtpu{XXXXYYYY}".
  The 8-char hex suffix encodes pipe FD numbers used for control IPC.
  Accesses pipes via /proc/<pid>/fd/<N>.
`, version)
}

func fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "tpu-checkpoint: "+format+"\n", args...)
	os.Exit(1)
}

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		usage()
		os.Exit(1)
	}

	var (
		pid        = -1
		action     string
		getState   bool
		timeoutSec = 180
	)

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--help", "-h":
			usage()
			os.Exit(0)
		case "--pid", "-p":
			i++
			if i >= len(args) {
				fatal("--pid requires a value")
			}
			var err error
			pid, err = strconv.Atoi(args[i])
			if err != nil || pid <= 0 {
				fatal("invalid PID: %s", args[i])
			}
		case "--action":
			i++
			if i >= len(args) {
				fatal("--action requires a value")
			}
			action = strings.ToLower(args[i])
		case "--get-state":
			getState = true
		case "--timeout", "-t":
			i++
			if i >= len(args) {
				fatal("--timeout requires a value")
			}
			var err error
			timeoutSec, err = strconv.Atoi(args[i])
			if err != nil || timeoutSec <= 0 {
				fatal("invalid timeout: %s", args[i])
			}
		default:
			fatal("unknown option: %s\nrun 'tpu-checkpoint --help' for usage", args[i])
		}
	}

	if pid <= 0 {
		fatal("--pid is required")
	}

	switch {
	case getState:
		state, err := tpucontrol.GetState(pid, timeoutSec)
		if err != nil {
			fatal("%v", err)
		}
		fmt.Println(state)

	case action == "checkpoint":
		t0 := time.Now()
		if err := tpucontrol.Checkpoint(pid, timeoutSec); err != nil {
			fatal("%v", err)
		}
		fmt.Printf("checkpoint complete (%.3fs)\n", time.Since(t0).Seconds())

	case action == "restore":
		t0 := time.Now()
		if err := tpucontrol.Restore(pid, timeoutSec); err != nil {
			fatal("%v", err)
		}
		fmt.Printf("restore complete (%.3fs)\n", time.Since(t0).Seconds())

	case action != "":
		fatal("unknown action: %s (supported: checkpoint, restore)", action)

	default:
		fatal("no operation specified (use --action checkpoint|restore or --get-state)\nrun 'tpu-checkpoint --help' for usage")
	}
}