import sys
import threading

from inspection_system.app.command_runner import launch_monitored_command, stream_command


def test_stream_command_emits_output_and_completion(tmp_path) -> None:
    output_lines: list[str] = []
    completed = {"code": None}
    done = threading.Event()

    stream_command(
        [sys.executable, "-c", "print('alpha'); print('beta')"],
        cwd=tmp_path,
        on_output=output_lines.append,
        on_complete=lambda code: (completed.__setitem__("code", code), done.set()),
    )

    assert done.wait(5)
    assert completed["code"] == 0
    assert output_lines == ["alpha\n", "beta\n"]


def test_launch_monitored_command_reports_exit_and_output(tmp_path) -> None:
    output_lines: list[str] = []
    completed = {"code": None}
    done = threading.Event()

    process = launch_monitored_command(
        [sys.executable, "-c", "print('ready'); raise SystemExit(3)"],
        cwd=tmp_path,
        on_output=output_lines.append,
        on_exit=lambda code: (completed.__setitem__("code", code), done.set()),
    )

    assert process.poll() is None or isinstance(process.poll(), int)
    assert done.wait(5)
    assert completed["code"] == 3
    assert output_lines == ["ready\n"]