"""Training subsystem split out of ``interactive_training.py``.

Phase 4 of the Beacon refactor pulls the long-lived helpers that have
clear responsibilities -- session logging today, more later -- into a
dedicated ``training`` package while keeping the historical import paths
working through re-exports in :mod:`inspection_system.app.interactive_training`.
"""
