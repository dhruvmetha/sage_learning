"""
Comprehensive logging management for auto-evaluation pipeline.
Provides structured logging for subprocesses and distributed evaluation.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, TextIO, Tuple
import json


class LogManager:
    """Manages all logging for the auto-evaluation pipeline."""
    
    def __init__(self, base_log_dir: str = "debug_logs", experiment_name: Optional[str] = None):
        """
        Initialize LogManager with timestamped experiment directory.
        
        Args:
            base_log_dir: Base directory for all logs
            experiment_name: Optional experiment name, otherwise uses timestamp
        """
        self.base_log_dir = Path(base_log_dir)
        
        # Create timestamped experiment directory
        if experiment_name:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.experiment_dir = self.base_log_dir / f"{experiment_name}_{timestamp}"
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.experiment_dir = self.base_log_dir / f"auto_eval_{timestamp}"
        
        # Create directory structure
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.inference_logs_dir = self.experiment_dir / "inference_servers"
        self.evaluation_logs_dir = self.experiment_dir / "evaluations"
        self.inference_logs_dir.mkdir(exist_ok=True)
        self.evaluation_logs_dir.mkdir(exist_ok=True)
        
        # Track open file handles for cleanup
        self._open_files: Dict[str, TextIO] = {}
        
        # Setup main pipeline logger
        self.main_log_path = self.experiment_dir / "auto_eval_pipeline.log"
        self._setup_main_logger()
        
        # Process monitoring
        self.process_monitor_path = self.experiment_dir / "process_monitor.log"
        self.process_states: Dict[str, Dict] = {}
        
        self.logger.info(f"LogManager initialized. Experiment directory: {self.experiment_dir}")
        
    def _setup_main_logger(self):
        """Setup the main pipeline logger with file output only (no console to avoid duplicates)."""
        self.logger = logging.getLogger(f"auto_eval_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler only - console logging handled by main script
        file_handler = logging.FileHandler(self.main_log_path)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate console logs
        self.logger.propagate = False
    
    def get_inference_log_files(self, server_id: str) -> Tuple[Path, Path]:
        """
        Get stdout and stderr log file paths for an inference server.
        
        Args:
            server_id: Unique identifier for the inference server
            
        Returns:
            Tuple of (stdout_path, stderr_path)
        """
        stdout_path = self.inference_logs_dir / f"{server_id}.out"
        stderr_path = self.inference_logs_dir / f"{server_id}.err"
        return stdout_path, stderr_path
    
    def get_evaluation_log_files(self, eval_id: str) -> Tuple[Path, Path]:
        """
        Get stdout and stderr log file paths for an evaluation process.
        
        Args:
            eval_id: Unique identifier for the evaluation process
            
        Returns:
            Tuple of (stdout_path, stderr_path)
        """
        stdout_path = self.evaluation_logs_dir / f"{eval_id}.out"
        stderr_path = self.evaluation_logs_dir / f"{eval_id}.err"
        return stdout_path, stderr_path
    
    def open_log_file(self, file_path: Path, mode: str = 'w') -> TextIO:
        """
        Open a log file and track it for cleanup.
        
        Args:
            file_path: Path to the log file
            mode: File opening mode
            
        Returns:
            File handle
        """
        file_handle = open(file_path, mode, buffering=1)  # Line buffered
        self._open_files[str(file_path)] = file_handle
        return file_handle
    
    def log_process_start(self, process_id: str, process_type: str, command: list, 
                         pid: int, log_files: Optional[Tuple[Path, Path]] = None):
        """
        Log the start of a subprocess.
        
        Args:
            process_id: Unique identifier for the process
            process_type: Type of process (inference_server, evaluation, etc.)
            command: Command that was executed
            pid: Process ID
            log_files: Optional tuple of (stdout_path, stderr_path)
        """
        start_time = time.time()
        process_info = {
            'process_id': process_id,
            'process_type': process_type,
            'command': command,
            'pid': pid,
            'start_time': start_time,
            'start_timestamp': datetime.fromtimestamp(start_time).isoformat(),
            'status': 'running',
            'log_files': {
                'stdout': str(log_files[0]) if log_files else None,
                'stderr': str(log_files[1]) if log_files else None
            }
        }
        
        self.process_states[process_id] = process_info
        self._write_process_monitor()
        
        self.logger.info(f"Started {process_type} process: {process_id} (PID: {pid})")
        self.logger.debug(f"Command: {' '.join(command)}")
        if log_files:
            self.logger.debug(f"Logs: stdout={log_files[0]}, stderr={log_files[1]}")
    
    def log_process_end(self, process_id: str, return_code: int, 
                       stdout: Optional[str] = None, stderr: Optional[str] = None):
        """
        Log the end of a subprocess.
        
        Args:
            process_id: Unique identifier for the process
            return_code: Process exit code
            stdout: Optional stdout content
            stderr: Optional stderr content
        """
        if process_id not in self.process_states:
            self.logger.warning(f"Process {process_id} not found in tracking")
            return
        
        end_time = time.time()
        process_info = self.process_states[process_id]
        
        process_info.update({
            'end_time': end_time,
            'end_timestamp': datetime.fromtimestamp(end_time).isoformat(),
            'duration': end_time - process_info['start_time'],
            'return_code': return_code,
            'status': 'completed' if return_code == 0 else 'failed'
        })
        
        self._write_process_monitor()
        
        if return_code == 0:
            self.logger.info(f"Process {process_id} completed successfully")
        else:
            self.logger.error(f"Process {process_id} failed with return code {return_code}")
            if stderr:
                self.logger.error(f"Error output: {stderr[:500]}...")  # Truncate long errors
    
    def _write_process_monitor(self):
        """Write current process states to monitor file."""
        with open(self.process_monitor_path, 'w') as f:
            json.dump(self.process_states, f, indent=2)
    
    def log_error(self, message: str, exc_info: bool = False):
        """Log an error message."""
        self.logger.error(message, exc_info=exc_info)
    
    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
    
    def tail_log_file(self, file_path: Path, lines: int = 50) -> str:
        """
        Get the last N lines from a log file.
        
        Args:
            file_path: Path to the log file
            lines: Number of lines to return
            
        Returns:
            Last N lines of the file
        """
        try:
            with open(file_path, 'r') as f:
                return ''.join(f.readlines()[-lines:])
        except FileNotFoundError:
            return f"Log file not found: {file_path}"
        except Exception as e:
            return f"Error reading log file {file_path}: {e}"
    
    def get_process_summary(self) -> Dict:
        """Get a summary of all tracked processes."""
        summary = {
            'total_processes': len(self.process_states),
            'running': sum(1 for p in self.process_states.values() if p['status'] == 'running'),
            'completed': sum(1 for p in self.process_states.values() if p['status'] == 'completed'),
            'failed': sum(1 for p in self.process_states.values() if p['status'] == 'failed'),
            'processes': self.process_states
        }
        return summary
    
    def cleanup(self):
        """Clean up resources and close open files."""
        self.logger.info("LogManager cleanup started")
        
        # Close all open files
        for file_path, file_handle in self._open_files.items():
            try:
                file_handle.close()
                self.logger.debug(f"Closed log file: {file_path}")
            except Exception as e:
                self.logger.error(f"Error closing file {file_path}: {e}")
        
        self._open_files.clear()
        
        # Final process monitor update
        self._write_process_monitor()
        
        # Log final summary
        summary = self.get_process_summary()
        self.logger.info(f"Final process summary: {summary['total_processes']} total, "
                        f"{summary['completed']} completed, {summary['failed']} failed")
        
        self.logger.info(f"All logs saved to: {self.experiment_dir}")
        self.logger.info("LogManager cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()