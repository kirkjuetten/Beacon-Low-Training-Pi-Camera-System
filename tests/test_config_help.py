"""Tests for inspection_system.app.config_help and the --explain CLI."""
from __future__ import annotations

import io
from contextlib import redirect_stdout

from inspection_system.__main__ import main as cli_main
from inspection_system.app import config_help
from inspection_system.app.config_validation import validate_config


def test_every_entry_has_required_fields() -> None:
    entries = config_help.all_entries()
    assert len(entries) >= 8
    for e in entries:
        assert e.key and "." in e.key, e.key
        assert e.summary
        assert e.valid
        assert e.starting_point
        assert e.symptom


def test_get_entry_known_and_unknown() -> None:
    assert config_help.get_entry("inspection.threshold_value") is not None
    assert config_help.get_entry("does.not.exist") is None


def test_format_entry_includes_all_sections() -> None:
    entry = config_help.get_entry("io.mode")
    assert entry is not None
    out = config_help.format_entry(entry)
    assert "io.mode" in out
    assert "What it does" in out
    assert "Valid values" in out
    assert "Start with" in out
    assert "If wrong" in out


def test_format_all_concatenates_entries() -> None:
    out = config_help.format_all()
    assert "inspection.threshold_value" in out
    assert "io.modbus.slave_id" in out
    # Entries separated by blank lines
    assert "\n\n" in out


def test_hint_for_known_and_unknown() -> None:
    assert "--explain inspection.threshold_value" in config_help.hint_for(
        "inspection.threshold_value"
    )
    assert config_help.hint_for("does.not.exist") == ""


def test_validation_messages_carry_hints() -> None:
    issues = validate_config(
        {
            "inspection": {"threshold_value": 999},
            "io": {"mode": "bogus"},
        }
    )
    joined = "\n".join(issues)
    assert "--explain inspection.threshold_value" in joined
    assert "--explain io.mode" in joined


def _run_cli(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(list(argv))
    return rc, buf.getvalue()


def test_cli_explain_known_key() -> None:
    rc, out = _run_cli("--explain", "inspection.threshold_value")
    assert rc == 0
    assert "inspection.threshold_value" in out
    assert "Start with" in out


def test_cli_explain_all() -> None:
    rc, out = _run_cli("--explain", "all")
    assert rc == 0
    assert "io.mode" in out
    assert "inspection.roi.width" in out


def test_cli_explain_unknown_key_lists_known() -> None:
    rc, out = _run_cli("--explain", "no.such.key")
    assert rc == 1
    assert "Unknown config key" in out
    assert "inspection.threshold_value" in out


def test_cli_no_args_prints_help() -> None:
    rc, out = _run_cli()
    assert rc == 0
    assert "--self-test" in out
    assert "--explain" in out
