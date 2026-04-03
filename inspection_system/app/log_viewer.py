#!/usr/bin/env python3
"""
Training Log Viewer for Beacon Inspection System

Displays and analyzes training session logs.
"""

import json
from pathlib import Path
from typing import Dict, List
import argparse


def load_training_logs(log_dir: Path) -> List[Dict]:
    """Load all training log files."""
    logs = []
    if not log_dir.exists():
        return logs

    for log_file in log_dir.glob("training_session_*.log"):
        session_logs = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('==='):
                    # Parse log line: [timestamp] STATUS -> FEEDBACK | filename | metrics
                    try:
                        parts = line.strip().split(' | ')
                        if len(parts) >= 4:
                            time_part = parts[0].strip('[]')
                            status_feedback = parts[1].split(' -> ')
                            if len(status_feedback) == 2:
                                status, feedback = status_feedback
                                filename = parts[2]
                                metrics_str = ' | '.join(parts[3:])

                                session_logs.append({
                                    'timestamp': time_part,
                                    'status': status,
                                    'feedback': feedback,
                                    'filename': filename,
                                    'metrics': metrics_str,
                                    'session': log_file.name
                                })
                    except:
                        continue
        logs.extend(session_logs)

    return sorted(logs, key=lambda x: x['timestamp'])


def analyze_logs(logs: List[Dict]) -> Dict:
    """Analyze training logs for insights."""
    if not logs:
        return {}

    total_samples = len(logs)
    approved = len([l for l in logs if l['feedback'] == 'APPROVE'])
    rejected = len([l for l in logs if l['feedback'] == 'REJECT'])
    reviewed = len([l for l in logs if l['feedback'] == 'REVIEW'])

    # Calculate approval rate
    approval_rate = approved / total_samples if total_samples > 0 else 0

    # Group by session
    sessions = {}
    for log in logs:
        session = log['session']
        if session not in sessions:
            sessions[session] = []
        sessions[session].append(log)

    return {
        'total_samples': total_samples,
        'approved': approved,
        'rejected': rejected,
        'reviewed': reviewed,
        'approval_rate': approval_rate,
        'sessions': len(sessions),
        'session_details': sessions
    }


def display_log_summary(logs: List[Dict]):
    """Display a summary of training logs."""
    if not logs:
        print("No training logs found.")
        return

    analysis = analyze_logs(logs)

    print("=== Training Log Summary ===")
    print(f"Total training samples: {analysis['total_samples']}")
    print(f"Approved: {analysis['approved']} ({analysis['approval_rate']:.1%})")
    print(f"Rejected: {analysis['rejected']}")
    print(f"Flagged for review: {analysis['reviewed']}")
    print(f"Training sessions: {analysis['sessions']}")
    print()


def display_recent_logs(logs: List[Dict], limit: int = 10):
    """Display recent training decisions."""
    if not logs:
        print("No training logs found.")
        return

    print(f"=== Recent Training Decisions (last {min(limit, len(logs))}) ===")
    for log in logs[-limit:]:
        print(f"[{log['timestamp']}] {log['status']} -> {log['feedback']}")
        print(f"  File: {log['filename']}")
        print(f"  Details: {log['metrics']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="View Beacon Inspection training logs")
    parser.add_argument("--log-dir", type=Path, default=Path("inspection_system/logs"),
                       help="Directory containing training logs")
    parser.add_argument("--recent", type=int, default=10,
                       help="Number of recent logs to display")
    parser.add_argument("--summary-only", action="store_true",
                       help="Show only summary, not individual logs")

    args = parser.parse_args()

    logs = load_training_logs(args.log_dir)

    display_log_summary(logs)

    if not args.summary_only:
        display_recent_logs(logs, args.recent)


if __name__ == "__main__":
    main()