import atexit
import os
import sys


class _TeeTextStream:
    """Write text to both the original stream and a log file stream."""

    def __init__(self, primary_stream, file_stream):
        self._primary_stream = primary_stream
        self._file_stream = file_stream

    def write(self, data):
        self._primary_stream.write(data)
        self._file_stream.write(data)
        return len(data)

    def flush(self):
        self._primary_stream.flush()
        self._file_stream.flush()

    def __getattr__(self, name):
        return getattr(self._primary_stream, name)


class RunLogCapture:
    def __init__(self, stdout_path, stderr_path, orig_stdout, orig_stderr, stdout_file, stderr_file):
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self._orig_stdout = orig_stdout
        self._orig_stderr = orig_stderr
        self._stdout_file = stdout_file
        self._stderr_file = stderr_file
        self._closed = False

    def close(self):
        if self._closed:
            return

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

        self._stdout_file.close()
        self._stderr_file.close()
        self._closed = True


def start_run_log_capture(root_dir, prefix):
    """Mirror stdout/stderr into files under a run directory.

    The file names follow:
      - <prefix>_<SLURM_JOB_ID>.out/.err when running under Slurm
      - <prefix>_pid<PID>.out/.err otherwise
    """
    os.makedirs(root_dir, exist_ok=True)

    job_id = os.getenv("SLURM_JOB_ID")
    suffix = str(job_id) if job_id else f"pid{os.getpid()}"

    stdout_path = os.path.join(root_dir, f"{prefix}_{suffix}.out")
    stderr_path = os.path.join(root_dir, f"{prefix}_{suffix}.err")

    stdout_file = open(stdout_path, "a", encoding="utf-8", buffering=1)
    stderr_file = open(stderr_path, "a", encoding="utf-8", buffering=1)

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    sys.stdout = _TeeTextStream(orig_stdout, stdout_file)
    sys.stderr = _TeeTextStream(orig_stderr, stderr_file)

    capture = RunLogCapture(
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        orig_stdout=orig_stdout,
        orig_stderr=orig_stderr,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
    )

    atexit.register(capture.close)
    return capture
