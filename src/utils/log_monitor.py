#!/usr/bin/env python3
"""
Log monitoring utilities for auto-evaluation pipeline debugging.
Provides tools to view, analyze, and monitor logs in real-time.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import sys


class LogMonitor:
    """Monitor and analyze auto-evaluation logs."""
    
    def __init__(self, log_dir: Path):
        """
        Initialize LogMonitor with a log directory.
        
        Args:
            log_dir: Path to the experiment log directory
        """
        self.log_dir = Path(log_dir)
        if not self.log_dir.exists():
            raise ValueError(f"Log directory does not exist: {log_dir}")
        
        self.main_log = self.log_dir / "auto_eval_pipeline.log"
        self.process_monitor = self.log_dir / "process_monitor.log"
        self.inference_dir = self.log_dir / "inference_servers"
        self.evaluation_dir = self.log_dir / "evaluations"
    
    def list_experiments(self, base_dir: str = "debug_logs") -> List[Path]:
        """List all available experiment directories."""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        
        experiments = []
        for item in base_path.iterdir():
            if item.is_dir() and (item / "auto_eval_pipeline.log").exists():
                experiments.append(item)
        
        return sorted(experiments, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def get_process_status(self) -> Dict:
        """Get current process status from process monitor."""
        if not self.process_monitor.exists():
            return {"error": "Process monitor file not found"}
        
        try:
            with open(self.process_monitor, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to read process monitor: {e}"}
    
    def tail_main_log(self, lines: int = 50) -> str:
        """Get the last N lines from the main pipeline log."""
        return self._tail_file(self.main_log, lines)
    
    def tail_inference_log(self, server_id: str, stream: str = "out", lines: int = 50) -> str:
        """
        Get the last N lines from an inference server log.
        
        Args:
            server_id: Inference server ID
            stream: "out" or "err" for stdout or stderr
            lines: Number of lines to show
        """
        log_file = self.inference_dir / f"{server_id}.{stream}"
        return self._tail_file(log_file, lines)
    
    def tail_evaluation_log(self, eval_id: str, stream: str = "out", lines: int = 50) -> str:
        """
        Get the last N lines from an evaluation log.
        
        Args:
            eval_id: Evaluation process ID
            stream: "out" or "err" for stdout or stderr
            lines: Number of lines to show
        """
        log_file = self.evaluation_dir / f"{eval_id}.{stream}"
        return self._tail_file(log_file, lines)
    
    def _tail_file(self, file_path: Path, lines: int) -> str:
        """Get the last N lines from a file."""
        if not file_path.exists():
            return f"Log file not found: {file_path}"
        
        try:
            with open(file_path, 'r') as f:
                all_lines = f.readlines()
                return ''.join(all_lines[-lines:])
        except Exception as e:
            return f"Error reading {file_path}: {e}"
    
    def watch_file(self, file_path: Path, follow: bool = True):
        """Watch a file in real-time (like tail -f)."""
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
        
        try:
            if follow:
                # Use system tail -f for real-time following
                subprocess.run(['tail', '-f', str(file_path)])
            else:
                # Just show current content
                with open(file_path, 'r') as f:
                    print(f.read())
        except KeyboardInterrupt:
            print("\nStopped watching file")
        except Exception as e:
            print(f"Error watching file: {e}")
    
    def search_logs(self, pattern: str, include_inference: bool = True, 
                   include_evaluation: bool = True) -> Dict[str, List[str]]:
        """
        Search for a pattern across all log files.
        
        Args:
            pattern: Search pattern (regex supported)
            include_inference: Include inference server logs
            include_evaluation: Include evaluation logs
            
        Returns:
            Dictionary mapping file paths to matching lines
        """
        results = {}
        
        # Search main log
        main_matches = self._grep_file(self.main_log, pattern)
        if main_matches:
            results[str(self.main_log)] = main_matches
        
        # Search inference logs
        if include_inference and self.inference_dir.exists():
            for log_file in self.inference_dir.glob("*.out"):
                matches = self._grep_file(log_file, pattern)
                if matches:
                    results[str(log_file)] = matches
            
            for log_file in self.inference_dir.glob("*.err"):
                matches = self._grep_file(log_file, pattern)
                if matches:
                    results[str(log_file)] = matches
        
        # Search evaluation logs
        if include_evaluation and self.evaluation_dir.exists():
            for log_file in self.evaluation_dir.glob("*.out"):
                matches = self._grep_file(log_file, pattern)
                if matches:
                    results[str(log_file)] = matches
            
            for log_file in self.evaluation_dir.glob("*.err"):
                matches = self._grep_file(log_file, pattern)
                if matches:
                    results[str(log_file)] = matches
        
        return results
    
    def _grep_file(self, file_path: Path, pattern: str) -> List[str]:
        """Search for pattern in a file and return matching lines."""
        if not file_path.exists():
            return []
        
        try:
            result = subprocess.run(
                ['grep', '-n', pattern, str(file_path)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
            return []
        except Exception:
            # Fallback to Python implementation
            try:
                import re
                matches = []
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if re.search(pattern, line):
                            matches.append(f"{line_num}:{line.rstrip()}")
                return matches
            except Exception:
                return []
    
    def generate_summary(self) -> str:
        """Generate a summary report of the experiment."""
        summary = []
        summary.append(f"Log Directory: {self.log_dir}")
        summary.append(f"Experiment Time: {self.log_dir.name}")
        summary.append("")
        
        # Process status
        status = self.get_process_status()
        if "error" not in status:
            total = len(status)
            running = sum(1 for p in status.values() if p.get('status') == 'running')
            completed = sum(1 for p in status.values() if p.get('status') == 'completed')
            failed = sum(1 for p in status.values() if p.get('status') == 'failed')
            
            summary.append(f"Process Summary:")
            summary.append(f"  Total: {total}")
            summary.append(f"  Running: {running}")
            summary.append(f"  Completed: {completed}")
            summary.append(f"  Failed: {failed}")
        else:
            summary.append(f"Process Status: {status['error']}")
        
        summary.append("")
        
        # Log files
        if self.inference_dir.exists():
            inference_logs = list(self.inference_dir.glob("*.out"))
            summary.append(f"Inference Logs: {len(inference_logs)} servers")
        
        if self.evaluation_dir.exists():
            eval_logs = list(self.evaluation_dir.glob("*.out"))
            summary.append(f"Evaluation Logs: {len(eval_logs)} evaluations")
        
        summary.append("")
        summary.append("Recent Main Log:")
        summary.append(self.tail_main_log(10))
        
        return '\n'.join(summary)


def main():
    """CLI interface for log monitoring."""
    parser = argparse.ArgumentParser(description="Monitor auto-evaluation logs")
    parser.add_argument("--log-dir", type=str, help="Path to experiment log directory")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--status", action="store_true", help="Show process status")
    parser.add_argument("--tail", type=str, help="Tail a specific log file")
    parser.add_argument("--watch", type=str, help="Watch a log file in real-time")
    parser.add_argument("--search", type=str, help="Search pattern across logs")
    parser.add_argument("--summary", action="store_true", help="Generate experiment summary")
    parser.add_argument("--lines", type=int, default=50, help="Number of lines to show")
    
    args = parser.parse_args()
    
    if args.list:
        experiments = LogMonitor("").list_experiments()
        if experiments:
            print("Available experiments:")
            for exp in experiments:
                print(f"  {exp}")
        else:
            print("No experiments found in debug_logs/")
        return
    
    if not args.log_dir:
        # Try to find the latest experiment
        monitor = LogMonitor("")
        experiments = monitor.list_experiments()
        if experiments:
            args.log_dir = str(experiments[0])
            print(f"Using latest experiment: {args.log_dir}")
        else:
            print("No log directory specified and no experiments found")
            sys.exit(1)
    
    try:
        monitor = LogMonitor(args.log_dir)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if args.status:
        status = monitor.get_process_status()
        print(json.dumps(status, indent=2))
    
    elif args.tail:
        content = monitor.tail_main_log(args.lines) if args.tail == "main" else monitor._tail_file(Path(args.tail), args.lines)
        print(content)
    
    elif args.watch:
        file_path = Path(args.watch) if args.watch != "main" else monitor.main_log
        monitor.watch_file(file_path)
    
    elif args.search:
        results = monitor.search_logs(args.search)
        for file_path, matches in results.items():
            print(f"\n{file_path}:")
            for match in matches:
                print(f"  {match}")
    
    elif args.summary:
        print(monitor.generate_summary())
    
    else:
        print(monitor.generate_summary())


if __name__ == "__main__":
    main()