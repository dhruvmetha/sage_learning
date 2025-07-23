import subprocess
import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
import json

logger = logging.getLogger(__name__)

class SSHCoordinator:
    def __init__(self, ssh_host: str, ml4kp_path: str, eval_script_path: str, log_manager=None):
        self.ssh_host = ssh_host
        self.ml4kp_path = ml4kp_path
        self.eval_script_path = eval_script_path
        self.active_processes = {}  # {process_id: subprocess}
        self.log_manager = log_manager  # Optional log manager for enhanced logging
        
    def test_ssh_connection(self) -> bool:
        """Test SSH connection to westeros"""
        try:
            result = subprocess.run([
                'ssh', self.ssh_host, 'echo "SSH connection test successful"'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info(f"SSH connection to {self.ssh_host} successful")
                return True
            else:
                logger.error(f"SSH connection failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("SSH connection timeout")
            return False
        except Exception as e:
            logger.error(f"SSH connection error: {e}")
            return False
    
    def check_ml4kp_path(self) -> bool:
        """Check if ml4kp path exists on westeros"""
        try:
            result = subprocess.run([
                'ssh', self.ssh_host, f'test -d {self.ml4kp_path}'
            ], capture_output=True, timeout=10)
            
            if result.returncode == 0:
                logger.info(f"ML4KP path {self.ml4kp_path} exists on {self.ssh_host}")
                return True
            else:
                logger.error(f"ML4KP path {self.ml4kp_path} not found on {self.ssh_host}")
                return False
        except Exception as e:
            logger.error(f"Error checking ML4KP path: {e}")
            return False
    
    def run_eval_namo(self, model_name: str, env_set_name: str, env_configs: List[str], 
                      inference_host: str, inference_port: int, num_trials: int = 50) -> Optional[str]:
        """
        Run eval_namo.py on westeros for given environment set
        Returns process ID for tracking
        """
        # Build command line arguments for eval_namo.py
        env_configs_str = ' '.join(env_configs)
        endpoint = f"tcp://{inference_host}:{inference_port}"
        
        try:
            # Execute evaluation on westeros with correct Python path and arguments
            # Set DIRTMP_PATH environment variable as required by eval_namo
            ssh_cmd = [
                'ssh', self.ssh_host,
                f'cd {self.ml4kp_path} && DIRTMP_PATH={self.ml4kp_path} /common/users/dm1487/envs/mjxrl/bin/python {self.eval_script_path} --model custom_walled_envs/jun22/random_start_random_goal_single_obstacle_room_2_200k_halfrad --num_trials {num_trials} --num_processes 1 --object_strategy 1 --endpoint {endpoint} --env_configs {env_configs_str}'
            ]
            
            # Create evaluation process ID
            process_id = f"{model_name}_{env_set_name}"
            
            # Setup logging if log_manager is available
            if self.log_manager:
                stdout_path, stderr_path = self.log_manager.get_evaluation_log_files(process_id)
                stdout_file = self.log_manager.open_log_file(stdout_path, 'w')
                stderr_file = self.log_manager.open_log_file(stderr_path, 'w')
                
                self.log_manager.log_info(f"Starting evaluation: {model_name} on {env_set_name} environments")
                self.log_manager.log_info(f"Command: {' '.join(ssh_cmd)}")
                self.log_manager.log_info(f"Logs: {stdout_path}, {stderr_path}")
                
                # Start process with file logging
                process = subprocess.Popen(
                    ssh_cmd, 
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True
                )
                
                # Log process start
                self.log_manager.log_process_start(
                    process_id=process_id,
                    process_type="ssh_evaluation",
                    command=ssh_cmd,
                    pid=process.pid,
                    log_files=(stdout_path, stderr_path)
                )
            else:
                # Fallback to old logging method
                logger.info(f"Starting evaluation: {model_name} on {env_set_name} environments")
                logger.info(f"Command: {' '.join(ssh_cmd)}")
                
                process = subprocess.Popen(
                    ssh_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            # Update process_id to include PID for uniqueness
            process_id = f"{model_name}_{env_set_name}_{process.pid}"
            self.active_processes[process_id] = {
                'process': process,
                'model_name': model_name,
                'env_set_name': env_set_name,
                'started_at': time.time()
            }
            
            logger.info(f"Started evaluation process {process_id}")
            return process_id
            
        except Exception as e:
            logger.error(f"Error running eval_namo: {e}")
            return None
    
    def _create_temp_eval_script(self, env_configs: List[str], inference_host: str, 
                                inference_port: int, num_trials: int) -> str:
        """Create a temporary eval_namo.py script with modified environment list"""
        
        # Read the original eval_namo.py from westeros (since that's the one that actually works)
        try:
            result = subprocess.run([
                'ssh', self.ssh_host, f'cat {self.ml4kp_path}/executables/utils/eval_namo.py'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"Failed to read original eval_namo.py: {result.stderr}")
                return None
                
            content = result.stdout
            
        except Exception as e:
            logger.error(f"Error reading original eval_namo.py: {e}")
            return None
        
        # Replace the environment list and endpoint
        env_list_str = repr(env_configs)
        
        # Replace the hardcoded environment list (from the actual westeros version)
        content = content.replace(
            "set_xml_files = ['env_config_182936', 'env_config_182741', 'env_config_182884', 'env_config_182782', 'env_config_182930', 'env_config_182945', 'env_config_182959', 'env_config_182799', 'env_config_182788']",
            f"set_xml_files = {env_list_str}"
        )
        
        # Replace the endpoint (from the actual westeros version)
        content = content.replace(
            'yaml_params["endpoint"] = "tcp://arrakis.cs.rutgers.edu:5557"',
            f'yaml_params["endpoint"] = "tcp://{inference_host}:{inference_port}"'
        )
        
        # Ensure object_strategy is set to 1 for ZMQ tests
        content = content.replace(
            'yaml_params["object_strategy"] = 1',
            'yaml_params["object_strategy"] = 1  # ZMQ/MCR mode'
        )
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.py', prefix='eval_namo_')
        with os.fdopen(temp_fd, 'w') as f:
            f.write(content)
        
        return temp_path
    
    def check_process_status(self, process_id: str) -> Optional[str]:
        """Check status of a running evaluation process"""
        if process_id not in self.active_processes:
            return None
        
        process_info = self.active_processes[process_id]
        process = process_info['process']
        
        # Check if process is still running
        poll_result = process.poll()
        
        if poll_result is None:
            return "running"
        elif poll_result == 0:
            return "completed"
        else:
            return "failed"
    
    def wait_for_process(self, process_id: str, timeout: int = 3600) -> Tuple[bool, str, str]:
        """
        Wait for a process to complete
        Returns (success, stdout, stderr)
        """
        if process_id not in self.active_processes:
            return False, "", "Process not found"
        
        process_info = self.active_processes[process_id]
        process = process_info['process']
        
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            success = process.returncode == 0
            
            # Remove from active processes
            del self.active_processes[process_id]
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Process {process_id} timed out after {timeout} seconds")
            process.kill()
            return False, "", "Process timed out"
    
    def _cleanup_remote_script(self, remote_script_path: str):
        """Clean up temporary script on westeros"""
        try:
            subprocess.run([
                'ssh', self.ssh_host, f'rm -f {remote_script_path}'
            ], capture_output=True, timeout=10)
        except Exception as e:
            logger.warning(f"Failed to cleanup remote script {remote_script_path}: {e}")
    
    def get_active_processes(self) -> Dict:
        """Get information about active processes"""
        status_info = {}
        for process_id, info in self.active_processes.items():
            status_info[process_id] = {
                'model_name': info['model_name'],
                'env_set_name': info['env_set_name'],
                'started_at': info['started_at'],
                'status': self.check_process_status(process_id)
            }
        return status_info
    
    def cleanup_all_processes(self):
        """Kill all active processes and cleanup"""
        for process_id in list(self.active_processes.keys()):
            try:
                process_info = self.active_processes[process_id]
                process = process_info['process']
                
                if process.poll() is None:  # Still running
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                
                
            except Exception as e:
                logger.error(f"Error cleaning up process {process_id}: {e}")
        
        self.active_processes.clear()
        logger.info("All processes cleaned up")

class ManualCoordinator:
    """Manual coordination - generates instruction files for user to execute"""
    
    def __init__(self, output_dir: str = "eval_instructions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.instruction_files = []
    
    def generate_instructions(self, evaluations: List[Dict]) -> str:
        """
        Generate instruction file for manual execution
        evaluations: List of {model_name, env_set_name, env_configs, host, port, num_trials}
        """
        timestamp = int(time.time())
        instruction_file = self.output_dir / f"eval_instructions_{timestamp}.txt"
        
        with open(instruction_file, 'w') as f:
            f.write("# Manual Evaluation Instructions\n")
            f.write(f"# Generated at: {time.ctime()}\n\n")
            f.write("# Step 1: SSH to westeros\n")
            f.write("ssh westeros.cs.rutgers.edu\n\n")
            f.write("# Step 2: Navigate to ML4KP directory\n")
            f.write("cd /common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp\n\n")
            
            for i, eval_info in enumerate(evaluations):
                f.write(f"# Step {i+3}: Run evaluation for {eval_info['model_name']} - {eval_info['env_set_name']}\n")
                f.write(f"# Inference server: {eval_info['host']}:{eval_info['port']}\n")
                f.write(f"# Environment configs: {eval_info['env_configs']}\n")
                
                # Create modified eval command with correct Python path
                cmd = (f"/common/users/dm1487/envs/mjxrl/bin/python executables/utils/eval_namo.py "
                      f"--model custom_walled_envs/jun22/random_start_random_goal_single_obstacle_room_2_200k_halfrad "
                      f"--num_trials {eval_info['num_trials']} --num_processes 1")
                
                f.write(f"{cmd}\n\n")
                f.write("# Note: You need to manually modify eval_namo.py to:\n")
                f.write(f"#   1. Set endpoint to 'tcp://{eval_info['host']}:{eval_info['port']}'\n")
                f.write(f"#   2. Set set_xml_files to {eval_info['env_configs']}\n\n")
        
        self.instruction_files.append(instruction_file)
        logger.info(f"Generated instruction file: {instruction_file}")
        return str(instruction_file)