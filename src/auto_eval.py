from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import subprocess
import time
import signal
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
import json
import queue
import threading
from dataclasses import dataclass

from utils.gpu_manager import ResourceManager
from utils.ssh_coordinator import SSHCoordinator, ManualCoordinator  
from utils.results_aggregator import ResultsAggregator
from utils.log_manager import LogManager

# Set up basic logging (LogManager will handle pipeline logging to files)
logging.basicConfig(level=logging.INFO, format='%(asctime)s[%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationJob:
    """Represents a single inference+evaluation job"""
    job_id: Optional[str]
    model_run_path: str
    env_set_name: Optional[str]  # None for regular inference (handles all env sets)
    is_split_inference: bool
    obstacle_run_path: Optional[str] = None
    goal_run_path: Optional[str] = None
    results_suffix: Optional[str] = None
    
    def __post_init__(self):
        if self.job_id is None:
            # Generate job_id from model name and env_set
            model_name = Path(self.model_run_path).name
            if self.env_set_name:
                self.job_id = f"{model_name}_{self.env_set_name}"
            else:
                self.job_id = model_name

class JobQueueManager:
    """Manages queued execution of inference+evaluation jobs with GPU limits"""
    
    def __init__(self, resource_manager, coordinator, log_manager, cfg):
        self.resource_manager = resource_manager
        self.coordinator = coordinator
        self.log_manager = log_manager
        self.cfg = cfg
        
        self.job_queue = queue.Queue()
        self.active_jobs = {}  # {job_id: {'server_info': tuple, 'eval_ids': list, 'job': EvaluationJob}}
        self.completed_jobs = []
        self.failed_jobs = []
        
        # Process tracking
        self.inference_servers = {}  # {server_id: process}
        self.evaluation_processes = {}  # {eval_id: process_info}
        
        self._stop_monitoring = False
        self._monitor_thread = None
    
    def add_job(self, job: EvaluationJob):
        """Add a job to the queue"""
        self.job_queue.put(job)
        self.log_manager.log_info(f"Added job to queue: {job.job_id}")
    
    def start_processing(self):
        """Start processing the job queue"""
        self.log_manager.log_info(f"Starting job queue processing. Queue size: {self.job_queue.qsize()}")
        
        # Start initial batch of jobs
        self._start_available_jobs()
        
        # Start monitoring thread
        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(target=self._monitor_jobs, daemon=True)
        self._monitor_thread.start()
    
    def _start_available_jobs(self):
        """Start jobs from queue up to resource limits"""
        max_concurrent = self.resource_manager.gpu_manager.max_gpus_to_use
        
        while (len(self.active_jobs) < max_concurrent and 
               not self.job_queue.empty()):
            
            try:
                job = self.job_queue.get_nowait()
                if self._start_job(job):
                    self.log_manager.log_info(f"Started job: {job.job_id}")
                else:
                    self.log_manager.log_warning(f"Failed to start job: {job.job_id}")
                    self.failed_jobs.append(job)
            except queue.Empty:
                break
    
    def _start_job(self, job: EvaluationJob) -> bool:
        """Start a single job (inference server + evaluation)"""
        try:
            # Start inference server
            server_info = self._start_inference_server_for_job(job)
            if not server_info:
                self.log_manager.log_error(f"Failed to start inference server for job {job.job_id}")
                return False
            
            server_id, gpu_id, port = server_info
            
            # Start evaluation(s)
            eval_ids = self._start_evaluation_for_job(job, server_info)
            if not eval_ids:
                self.log_manager.log_error(f"Failed to start evaluations for job {job.job_id}")
                # Clean up the inference server
                self._cleanup_inference_server(server_id)
                return False
            
            # Store job info
            self.active_jobs[job.job_id] = {
                'server_info': server_info,
                'eval_ids': eval_ids,
                'job': job,
                'started_at': time.time()
            }
            
            self.log_manager.log_info(f"Successfully started job {job.job_id} on GPU {gpu_id}, port {port}")
            return True
            
        except Exception as e:
            self.log_manager.log_error(f"Error starting job {job.job_id}: {e}", exc_info=True)
            return False
    
    def _start_inference_server_for_job(self, job: EvaluationJob) -> Optional[Tuple[str, int, int]]:
        """Start inference server for a specific job"""
        # Allocate resources
        resources = self.resource_manager.allocate_resources(f"inference_{job.job_id}")
        if resources is None:
            self.log_manager.log_error(f"Failed to allocate resources for job {job.job_id}")
            return None
        
        gpu_id, port = resources
        
        # Determine inference script type
        if job.is_split_inference:
            if not job.obstacle_run_path or not job.goal_run_path:
                self.log_manager.log_error("Split inference job missing obstacle_run_path or goal_run_path")
                self.resource_manager.release_resources(f"inference_{job.job_id}_{gpu_id}_{port}")
                return None
            inference_script = "src/split_inference_diffusion_zmq.py"
        else:
            inference_script = "src/inference_diffusion_zmq.py"
        
        # Create inference command
        cmd = [
            "/common/users/dm1487/envs/mjxrl/bin/python", inference_script
        ]
        
        if job.is_split_inference:
            cmd.extend([
                f"obstacle_run_path={job.obstacle_run_path}",
                f"goal_run_path={job.goal_run_path}",
                f"run_path={job.model_run_path}",
                f"zmq.port={port}",
                f"zmq.host=arrakis.cs.rutgers.edu"
            ])
            if job.results_suffix:
                cmd.append(f"+results_suffix={job.results_suffix}")
        else:
            cmd.extend([
                f"run_path={job.model_run_path}",
                f"zmq.port={port}",
                f"zmq.host=arrakis.cs.rutgers.edu"
            ])
        
        # Set environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Log GPU assignment for debugging
        self.log_manager.log_info(f"Job {job.job_id} assigned to GPU {gpu_id}")
        
        # Start server process
        try:
            server_id = f"inference_{job.job_id}_{gpu_id}_{port}"
            
            # Create log files
            log_dir = Path(self.log_manager.experiment_dir) / "inference_servers"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            stdout_file = log_dir / f"{server_id}.out"
            stderr_file = log_dir / f"{server_id}.err"
            
            with open(stdout_file, 'w') as out_f, open(stderr_file, 'w') as err_f:
                process = subprocess.Popen(
                    cmd, 
                    env=env, 
                    stdout=out_f, 
                    stderr=err_f,
                    cwd=Path.cwd()
                )
            
            # Store server info
            self.inference_servers[server_id] = {
                'process': process,
                'gpu_id': gpu_id,
                'port': port,
                'job_id': job.job_id,
                'started_at': time.time(),
                'stdout_file': stdout_file,
                'stderr_file': stderr_file
            }
            
            self.log_manager.log_process_start(server_id, "inference_server", cmd, process.pid, (stdout_file, stderr_file))
            
            # Wait a moment for server to start
            time.sleep(3)
            
            return server_id, gpu_id, port
            
        except Exception as e:
            self.log_manager.log_error(f"Error starting inference server for job {job.job_id}: {e}", exc_info=True)
            self.resource_manager.release_resources(f"inference_{job.job_id}_{gpu_id}_{port}")
            return None
    
    def _start_evaluation_for_job(self, job: EvaluationJob, server_info: Tuple[str, int, int]) -> List[str]:
        """Start evaluation process(es) for a job"""
        server_id, gpu_id, port = server_info
        eval_ids = []
        
        if job.is_split_inference:
            # For split inference, run evaluation for the specific environment set
            if job.env_set_name:
                eval_id = self._start_single_evaluation(job, server_info, job.env_set_name)
                if eval_id:
                    eval_ids.append(eval_id)
        else:
            # For regular inference, run evaluations for all environment sets
            for env_set_name in self.cfg.eval_environments.default_sets:
                eval_id = self._start_single_evaluation(job, server_info, env_set_name)
                if eval_id:
                    eval_ids.append(eval_id)
        
        return eval_ids
    
    def _start_single_evaluation(self, job: EvaluationJob, server_info: Tuple[str, int, int], env_set_name: str) -> Optional[str]:
        """Start a single evaluation for a specific environment set"""
        server_id, gpu_id, port = server_info
        model_name = Path(job.model_run_path).name
        
        env_configs = getattr(self.cfg.eval_environments, env_set_name, [])
        if not env_configs:
            self.log_manager.log_warning(f"No environments found for set: {env_set_name}")
            return None
        
        self.log_manager.log_info(f"Starting evaluation: {model_name} on {env_set_name} ({len(env_configs)} environments)")
        
        if self.cfg.coordination == "ssh":
            # Use SSH coordinator
            eval_id = self.coordinator.run_eval_namo(
                model_name=model_name,
                env_set_name=env_set_name,
                env_configs=env_configs,
                inference_host="arrakis.cs.rutgers.edu",
                inference_port=port,
                num_trials=self.cfg.num_trials
            )
            
            if eval_id:
                self.evaluation_processes[eval_id] = {
                    'model_name': model_name,
                    'env_set_name': env_set_name,
                    'server_id': server_id,
                    'job_id': job.job_id,
                    'started_at': time.time()
                }
                return eval_id
        else:
            # Manual coordination - just log what needs to be done
            self.log_manager.log_info(f"Manual coordination: User needs to run evaluation for {model_name} on {env_set_name}")
            self.log_manager.log_info(f"Inference server: arrakis.cs.rutgers.edu:{port}")
            self.log_manager.log_info(f"Environment configs: {env_configs}")
            
            # For manual coordination, return a dummy eval_id
            eval_id = f"manual_{job.job_id}_{env_set_name}"
            self.evaluation_processes[eval_id] = {
                'model_name': model_name,
                'env_set_name': env_set_name,
                'server_id': server_id,
                'job_id': job.job_id,
                'started_at': time.time(),
                'manual': True
            }
            return eval_id
        
        return None
    
    def _cleanup_inference_server(self, server_id: str):
        """Clean up a specific inference server"""
        if server_id in self.inference_servers:
            server_info = self.inference_servers[server_id]
            try:
                process = server_info['process']
                gpu_id = server_info['gpu_id']
                port = server_info['port']
                
                self.log_manager.log_info(f"Terminating inference server {server_id} on GPU {gpu_id}, port {port}")
                
                if process.poll() is None:
                    # Try graceful termination first
                    self.log_manager.log_info(f"Sending TERM signal to process {process.pid}")
                    process.terminate()
                    try:
                        process.wait(timeout=10)  # Wait up to 10 seconds
                        self.log_manager.log_info(f"Process {process.pid} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate gracefully
                        self.log_manager.log_warning(f"Force killing process {process.pid}")
                        process.kill()
                        process.wait(timeout=5)
                        self.log_manager.log_info(f"Process {process.pid} killed")
                else:
                    self.log_manager.log_info(f"Process {process.pid} already terminated")
                
                # Release resources
                allocation_id = f"inference_{server_info['job_id']}_{gpu_id}_{port}"
                self.resource_manager.release_resources(allocation_id)
                
                # Log process end
                self.log_manager.log_process_end(server_id, process.returncode)
                
                del self.inference_servers[server_id]
                self.log_manager.log_info(f"Successfully cleaned up inference server {server_id}")
                
            except Exception as e:
                self.log_manager.log_error(f"Error cleaning up inference server {server_id}: {e}")
                # Still try to release resources even if process cleanup failed
                try:
                    gpu_id = server_info['gpu_id']
                    port = server_info['port']
                    allocation_id = f"inference_{server_info['job_id']}_{gpu_id}_{port}"
                    self.resource_manager.release_resources(allocation_id)
                    del self.inference_servers[server_id]
                except:
                    pass
    
    def _monitor_jobs(self):
        """Monitor running jobs and start new ones when resources become available"""
        while not self._stop_monitoring:
            try:
                # Check for completed jobs
                completed_job_ids = self._check_completed_jobs()
                
                # Clean up completed jobs and start new ones
                for job_id in completed_job_ids:
                    self._cleanup_completed_job(job_id)
                
                # Start new jobs if resources available
                if completed_job_ids:
                    self._start_available_jobs()
                
                # Sleep before next check
                time.sleep(5)
                
            except Exception as e:
                self.log_manager.log_error(f"Error in job monitoring: {e}", exc_info=True)
                time.sleep(10)  # Longer sleep on error
    
    def _check_completed_jobs(self) -> List[str]:
        """Check which jobs have completed and return their job_ids"""
        completed_job_ids = []
        
        for job_id, job_info in list(self.active_jobs.items()):
            job = job_info['job']
            eval_ids = job_info['eval_ids']
            server_info = job_info['server_info']
            runtime = time.time() - job_info['started_at']
            
            # Don't check for completion too early - give processes time to start
            min_runtime = 30  # Minimum 30 seconds before checking completion
            if runtime < min_runtime:
                continue
            
            # Check if all evaluations for this job have completed
            all_evals_completed = True
            for eval_id in eval_ids:
                if not self._is_evaluation_completed(eval_id):
                    all_evals_completed = False
                    break
            
            if all_evals_completed:
                self.log_manager.log_info(f"Job {job_id} completed - all evaluations finished (runtime: {runtime:.0f}s)")
                self.log_manager.log_info(f"Will terminate inference server for job {job_id}")
                completed_job_ids.append(job_id)
            # No timeout - let jobs run indefinitely
        
        return completed_job_ids
    
    def _is_evaluation_completed(self, eval_id: str) -> bool:
        """Check if a specific evaluation has completed"""
        if eval_id not in self.evaluation_processes:
            return True  # Already cleaned up, consider completed
        
        eval_info = self.evaluation_processes[eval_id]
        
        # For manual coordination, evaluations never complete automatically
        if eval_info.get('manual', False):
            return False
        
        # For SSH coordination, check with the coordinator
        if self.cfg.coordination == "ssh":
            try:
                # Use the correct SSH coordinator method
                status = self.coordinator.check_process_status(eval_id)
                
                if status is None:
                    # Process not found, consider completed
                    self.log_manager.log_info(f"Evaluation {eval_id} completed (process not found)")
                    if eval_id in self.evaluation_processes:
                        eval_info = self.evaluation_processes[eval_id]
                        self.log_manager.log_info(f"Evaluation {eval_id} for job {eval_info.get('job_id')} finished - marking for inference server cleanup")
                        del self.evaluation_processes[eval_id]
                    return True
                elif status == "completed":
                    # Process completed
                    self.log_manager.log_info(f"Evaluation {eval_id} completed")
                    if eval_id in self.evaluation_processes:
                        eval_info = self.evaluation_processes[eval_id]
                        self.log_manager.log_info(f"Evaluation {eval_id} for job {eval_info.get('job_id')} finished - marking for inference server cleanup")
                        del self.evaluation_processes[eval_id]
                    return True
                else:
                    # Process still running
                    return False
                    
            except Exception as e:
                self.log_manager.log_error(f"Error checking evaluation {eval_id}: {e}")
                # Don't consider it completed on error - let it continue running
                return False
        
        return False
    
    def _cleanup_completed_job(self, job_id: str):
        """Clean up resources for a completed job"""
        if job_id in self.active_jobs:
            job_info = self.active_jobs[job_id]
            
            # Stop inference server if running
            server_info = job_info.get('server_info')
            if server_info:
                server_id, gpu_id, port = server_info
                self._cleanup_inference_server(server_id)
            
            # Clean up any remaining evaluation processes
            eval_ids = job_info.get('eval_ids', [])
            for eval_id in eval_ids:
                if eval_id in self.evaluation_processes:
                    del self.evaluation_processes[eval_id]
            
            # Move to completed
            self.completed_jobs.append(job_info['job'])
            del self.active_jobs[job_id]
            
            self.log_manager.log_info(f"Cleaned up completed job: {job_id}")
    
    def wait_for_completion(self, timeout: int = None):
        """Wait for all jobs to complete"""
        self.log_manager.log_info("Waiting for all jobs to complete...")
        
        # For manual coordination, generate instructions and wait for user input
        if self.cfg.coordination == "manual":
            self._handle_manual_coordination()
            return
        
        while (self.active_jobs or not self.job_queue.empty()):
            time.sleep(5)
            
            # Log progress
            active_count = len(self.active_jobs)
            queue_size = self.job_queue.qsize()
            completed_count = len(self.completed_jobs)
            
            self.log_manager.log_info(f"Job status - Active: {active_count}, Queued: {queue_size}, Completed: {completed_count}")
    
    def _handle_manual_coordination(self):
        """Handle manual coordination mode"""
        self.log_manager.log_info("Manual coordination mode - generating instructions...")
        
        # Generate instructions for all jobs (active + queued)
        all_jobs = []
        all_jobs.extend([job_info['job'] for job_info in self.active_jobs.values()])
        
        # Add queued jobs
        queued_jobs = []
        while not self.job_queue.empty():
            try:
                job = self.job_queue.get_nowait()
                queued_jobs.append(job)
            except queue.Empty:
                break
        all_jobs.extend(queued_jobs)
        
        # Generate instructions
        evaluations = []
        for job in all_jobs:
            server_info = None
            # For active jobs, get server info
            if job.job_id in self.active_jobs:
                server_info = self.active_jobs[job.job_id]['server_info']
            
            if job.is_split_inference and job.env_set_name:
                # Split inference - one job per env set
                env_configs = getattr(self.cfg.eval_environments, job.env_set_name, [])
                if env_configs:
                    evaluations.append({
                        'model_name': Path(job.model_run_path).name,
                        'env_set_name': job.env_set_name,
                        'env_configs': env_configs,
                        'host': 'arrakis.cs.rutgers.edu',
                        'port': server_info[2] if server_info else 'TBD',
                        'num_trials': self.cfg.num_trials,
                        'job_id': job.job_id
                    })
            else:
                # Regular inference - all env sets
                for env_set_name in self.cfg.eval_environments.default_sets:
                    env_configs = getattr(self.cfg.eval_environments, env_set_name, [])
                    if env_configs:
                        evaluations.append({
                            'model_name': Path(job.model_run_path).name,
                            'env_set_name': env_set_name,
                            'env_configs': env_configs,
                            'host': 'arrakis.cs.rutgers.edu',
                            'port': server_info[2] if server_info else 'TBD',
                            'num_trials': self.cfg.num_trials,
                            'job_id': job.job_id
                        })
        
        # Generate instruction file
        instruction_file = self.coordinator.generate_instructions(evaluations)
        self.log_manager.log_info(f"Generated manual instructions: {instruction_file}")
        self.log_manager.log_info("Please execute the instructions manually, then press Enter to continue...")
        input()  # Wait for user to complete manual evaluations
        
        # Mark all jobs as completed for manual mode
        for job in all_jobs:
            if job.job_id in self.active_jobs:
                self.completed_jobs.append(job)
                del self.active_jobs[job.job_id]
            else:
                self.completed_jobs.append(job)
    
    def stop_monitoring(self):
        """Stop the job monitoring thread"""
        self._stop_monitoring = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=30)
    
    def cleanup_all(self):
        """Clean up all active jobs and resources"""
        self.log_manager.log_info("Cleaning up all active jobs...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Clean up active jobs
        for job_id in list(self.active_jobs.keys()):
            self._cleanup_completed_job(job_id)

class AutoEvaluationPipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        # Initialize comprehensive logging
        experiment_name = getattr(cfg, 'experiment_name', None)
        log_dir = getattr(cfg, 'log_dir', 'debug_logs')
        self.log_manager = LogManager(base_log_dir=log_dir, experiment_name=experiment_name)
        
        self.resource_manager = ResourceManager(
            available_gpus=cfg.available_gpus,
            max_gpus_to_use=cfg.max_gpus_to_use,
            start_port=cfg.start_port
        )
        
        # Initialize coordination method
        if cfg.coordination == "ssh":
            self.coordinator = SSHCoordinator(
                ssh_host=cfg.ssh.host,
                ml4kp_path=cfg.ssh.ml4kp_path,
                eval_script_path=cfg.ssh.eval_script_path,
                log_manager=self.log_manager  # Pass log_manager to coordinator
            )
        else:
            self.coordinator = ManualCoordinator()
        
        self.results_aggregator = ResultsAggregator(cfg.results_dir)
        
        # Initialize job queue manager
        self.job_queue_manager = JobQueueManager(
            self.resource_manager, 
            self.coordinator, 
            self.log_manager, 
            cfg
        )
        
        # Legacy process tracking (will be moved to JobQueueManager)
        self.inference_servers = {}  # {server_id: process}
        self.evaluation_processes = {}  # {eval_id: process_info}
        
        # Set up signal handlers for cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.log_manager.log_info("AutoEvaluationPipeline initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.log_manager.log_info("Received shutdown signal, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def discover_model_runs(self) -> List[str]:
        """Discover recently completed training runs"""
        if self.cfg.model_runs:
            # Use explicitly specified model runs
            self.log_manager.log_info(f"Using specified model runs: {self.cfg.model_runs}")
            return self.cfg.model_runs
        
        if not self.cfg.auto_detect_latest:
            self.log_manager.log_error("No model runs specified and auto-detection disabled")
            return []
        
        # Auto-detect latest training runs
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            self.log_manager.log_error("Outputs directory not found")
            return []
        
        # Find directories with completed training (have checkpoints)
        completed_runs = []
        
        # Sort by modification time (newest first)
        date_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], 
                          key=lambda x: x.stat().st_mtime, reverse=True)
        
        for date_dir in date_dirs:
            time_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()],
                             key=lambda x: x.stat().st_mtime, reverse=True)
            
            for time_dir in time_dirs:
                checkpoint_dir = time_dir / "checkpoints"
                if checkpoint_dir.exists():
                    # Check if has epoch checkpoint (not just last.ckpt)
                    epoch_checkpoints = list(checkpoint_dir.glob("epoch=*.ckpt"))
                    if epoch_checkpoints:
                        completed_runs.append(str(time_dir))
                        self.log_manager.log_info(f"Found completed training run: {time_dir}")
                        
                        if len(completed_runs) >= self.cfg.max_models_to_detect:
                            break
            
            if len(completed_runs) >= self.cfg.max_models_to_detect:
                break
        
        self.log_manager.log_info(f"Auto-detected {len(completed_runs)} completed training runs")
        return completed_runs
    
    def start_inference_server(self, model_run_path: str, is_split_inference: bool = False, obstacle_run_path: str = None, goal_run_path: str = None, results_suffix: str = None) -> Optional[Tuple[str, int, int]]:
        """
        Start inference server for a model run
        Returns (server_id, gpu_id, port) or None if failed
        """
        # Allocate resources
        resources = self.resource_manager.allocate_resources(f"inference_{Path(model_run_path).name}")
        if resources is None:
            self.log_manager.log_error(f"Failed to allocate resources for {model_run_path}")
            return None
        
        gpu_id, port = resources
        
        # Determine inference script type
        if is_split_inference:
            if not obstacle_run_path or not goal_run_path:
                self.log_manager.log_error("Split inference requires both obstacle_run_path and goal_run_path")
                return None
            inference_script = "src/split_inference_diffusion_zmq.py"
            self.log_manager.log_info(f"Using split inference with obstacle: {obstacle_run_path}, goal: {goal_run_path}")
        else:
            model_run_path_obj = Path(model_run_path)
            hydra_config_path = model_run_path_obj / ".hydra" / "config.yaml"
            
            if hydra_config_path.exists():
                # Check if this is a split model or single model
                with open(hydra_config_path, 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                
                # Use single model inference
                inference_script = "src/inference_diffusion_zmq.py"
            else:
                self.log_manager.log_warning(f"No hydra config found for {model_run_path}, using default inference script")
                inference_script = "src/inference_diffusion_zmq.py"
        
        # Create inference command
        cmd = [
            "/common/users/dm1487/envs/mjxrl/bin/python", inference_script
        ]
        
        if is_split_inference:
            cmd.extend([
                f"obstacle_run_path={obstacle_run_path}",
                f"goal_run_path={goal_run_path}",
                f"run_path={model_run_path}",
                f"zmq.port={port}",
                f"zmq.host=arrakis.cs.rutgers.edu"
            ])
            # Add results_suffix for split inference to differentiate difficulty levels
            if results_suffix:
                cmd.append(f"results_suffix={results_suffix}")
        else:
            cmd.extend([
                f"run_path={model_run_path}",
                f"zmq.port={port}",
                f"zmq.host=arrakis.cs.rutgers.edu"
            ])
        
        # Set GPU environment and preserve conda environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Log GPU assignment for debugging
        model_name = Path(model_run_path).name
        self.log_manager.log_info(f"Model {model_name} assigned to GPU {gpu_id}")
        
        # Clean environment to avoid shell initialization interference
        for key in ["BASH_ENV", "ENV", "BASH_FUNC_*"]:
            if key in env:
                del env[key]
        
        # Ensure we use the correct conda environment
        if "CONDA_DEFAULT_ENV" in env and env["CONDA_DEFAULT_ENV"] != "mjxrl":
            self.log_manager.log_warning(f"Expected mjxrl environment, but found {env['CONDA_DEFAULT_ENV']}")
        
        # Set conda environment variables explicitly
        env["CONDA_DEFAULT_ENV"] = "mjxrl"
        env["CONDA_PREFIX"] = "/common/users/dm1487/envs/mjxrl"
        
        # Force unbuffered Python output
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Keep the existing conda environment path
        conda_path = env.get("PATH", "")
        if "/common/users/dm1487/envs/mjxrl/bin" not in conda_path:
            env["PATH"] = f"/common/users/dm1487/envs/mjxrl/bin:{conda_path}"
        
        # Get log file paths for this server
        server_id = f"inference_{Path(model_run_path).name}_{gpu_id}_{port}"
        stdout_path, stderr_path = self.log_manager.get_inference_log_files(server_id)
        
        self.log_manager.log_info(f"Starting inference server: {' '.join(cmd)}")
        self.log_manager.log_info(f"GPU: {gpu_id}, Port: {port}")
        self.log_manager.log_info(f"Logs: {stdout_path}, {stderr_path}")
        
        # Debug: Check if script exists and python is accessible
        python_path = cmd[0]
        script_path = cmd[1]
        self.log_manager.log_debug(f"Python path: {python_path}")
        self.log_manager.log_debug(f"Script path: {script_path}")
        self.log_manager.log_debug(f"Working directory: {os.getcwd()}")
        self.log_manager.log_debug(f"Script exists: {os.path.exists(script_path)}")
        self.log_manager.log_debug(f"Python exists: {os.path.exists(python_path)}")
        
        try:
            # Open log files
            stdout_file = self.log_manager.open_log_file(stdout_path, 'w')
            stderr_file = self.log_manager.open_log_file(stderr_path, 'w')
            
            # Start inference server with file logging
            # Add unbuffered output to ensure immediate log writes
            process = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
                text=True,
                shell=False,  # Explicitly avoid shell
                cwd=os.getcwd(),  # Ensure correct working directory
                bufsize=0  # Unbuffered for immediate output
            )
            
            # Log process start
            self.log_manager.log_process_start(
                process_id=server_id,
                process_type="inference_server",
                command=cmd,
                pid=process.pid,
                log_files=(stdout_path, stderr_path)
            )
            
            # Wait a bit to see if it starts successfully
            time.sleep(5)
            if process.poll() is not None:
                # Process died, get return code and log failure
                return_code = process.returncode
                
                # Try to get any output from the log files
                stdout_file.flush()
                stderr_file.flush()
                
                try:
                    with open(stdout_path, 'r') as f:
                        stdout_content = f.read()
                    with open(stderr_path, 'r') as f:
                        stderr_content = f.read()
                    
                    self.log_manager.log_debug(f"Process stdout: {stdout_content[:500]}")
                    self.log_manager.log_debug(f"Process stderr: {stderr_content[:500]}")
                except Exception as e:
                    self.log_manager.log_debug(f"Could not read log files: {e}")
                
                self.log_manager.log_process_end(server_id, return_code)
                self.log_manager.log_error(f"Inference server failed to start with return code {return_code}")
                self.resource_manager.release_resources(f"inference_{Path(model_run_path).name}_{gpu_id}_{port}")
                return None
            
            self.inference_servers[server_id] = {
                'process': process,
                'model_run_path': model_run_path,
                'gpu_id': gpu_id,
                'port': port,
                'started_at': time.time(),
                'log_files': (stdout_path, stderr_path)
            }
            
            self.log_manager.log_info(f"Inference server started successfully: {server_id}")
            return server_id, gpu_id, port
            
        except Exception as e:
            self.log_manager.log_error(f"Error starting inference server: {e}", exc_info=True)
            self.resource_manager.release_resources(f"inference_{Path(model_run_path).name}_{gpu_id}_{port}")
            return None
    
    def run_evaluations(self, server_info: Tuple[str, int, int], model_run_path: str) -> List[str]:
        """
        Run evaluations for a model against configured environment sets
        Returns list of evaluation process IDs
        """
        server_id, _gpu_id, port = server_info
        model_name = Path(model_run_path).name
        
        # Get environment sets to evaluate
        env_sets = self.cfg.eval_environments.default_sets
        evaluation_ids = []
        
        for env_set_name in env_sets:
            env_configs = getattr(self.cfg.eval_environments, env_set_name, [])
            if not env_configs:
                self.log_manager.log_warning(f"No environments found for set: {env_set_name}")
                continue
            
            self.log_manager.log_info(f"Starting evaluation: {model_name} on {env_set_name} ({len(env_configs)} environments)")
            
            if self.cfg.coordination == "ssh":
                # Use SSH coordinator
                eval_id = self.coordinator.run_eval_namo(
                    model_name=model_name,
                    env_set_name=env_set_name,
                    env_configs=env_configs,
                    inference_host="arrakis.cs.rutgers.edu",
                    inference_port=port,
                    num_trials=self.cfg.num_trials
                )
                
                if eval_id:
                    evaluation_ids.append(eval_id)
                    self.evaluation_processes[eval_id] = {
                        'model_name': model_name,
                        'env_set_name': env_set_name,
                        'server_id': server_id,
                        'started_at': time.time()
                    }
            else:
                # Manual coordination - just log what needs to be done
                self.log_manager.log_info(f"Manual coordination: User needs to run evaluation for {model_name} on {env_set_name}")
                self.log_manager.log_info(f"Inference server: arrakis.cs.rutgers.edu:{port}")
                self.log_manager.log_info(f"Environment configs: {env_configs}")
        
        return evaluation_ids
    
    def _build_job_queue(self, model_runs: List[str], is_split: bool, obstacle_path: str = None, goal_path: str = None):
        """Build the job queue from model runs and environment sets"""
        job_count = 0
        
        if is_split:
            # For split inference, create separate jobs for each model+environment combination
            for model_run in model_runs:
                for env_set_name in self.cfg.eval_environments.default_sets:
                    results_suffix = f"split_results_{env_set_name}"
                    
                    job = EvaluationJob(
                        job_id=None,  # Will be auto-generated
                        model_run_path=model_run,
                        env_set_name=env_set_name,
                        is_split_inference=True,
                        obstacle_run_path=obstacle_path,
                        goal_run_path=goal_path,
                        results_suffix=results_suffix
                    )
                    
                    self.job_queue_manager.add_job(job)
                    job_count += 1
        else:
            # For regular inference, create one job per model (handles all environment sets)
            for model_run in model_runs:
                job = EvaluationJob(
                    job_id=None,  # Will be auto-generated
                    model_run_path=model_run,
                    env_set_name=None,  # None means handles all env sets
                    is_split_inference=False,
                    obstacle_run_path=None,
                    goal_run_path=None,
                    results_suffix=None
                )
                
                self.job_queue_manager.add_job(job)
                job_count += 1
        
        self.log_manager.log_info(f"Created {job_count} jobs for evaluation")
    
    def run_single_evaluation(self, server_info: Tuple[str, int, int], model_run_path: str, env_set_name: str) -> Optional[str]:
        """
        Run evaluation for a single model against a specific environment set
        Returns evaluation process ID or None if failed
        """
        server_id, _gpu_id, port = server_info
        model_name = Path(model_run_path).name
        
        env_configs = getattr(self.cfg.eval_environments, env_set_name, [])
        if not env_configs:
            self.log_manager.log_warning(f"No environments found for set: {env_set_name}")
            return None
        
        self.log_manager.log_info(f"Starting evaluation: {model_name} on {env_set_name} ({len(env_configs)} environments)")
        
        if self.cfg.coordination == "ssh":
            # Use SSH coordinator
            eval_id = self.coordinator.run_eval_namo(
                model_name=model_name,
                env_set_name=env_set_name,
                env_configs=env_configs,
                inference_host="arrakis.cs.rutgers.edu",
                inference_port=port,
                num_trials=self.cfg.num_trials
            )
            
            if eval_id:
                self.evaluation_processes[eval_id] = {
                    'model_name': model_name,
                    'env_set_name': env_set_name,
                    'server_id': server_id,
                    'started_at': time.time()
                }
                return eval_id
        else:
            # Manual coordination - just log what needs to be done
            self.log_manager.log_info(f"Manual coordination: User needs to run evaluation for {model_name} on {env_set_name}")
            self.log_manager.log_info(f"Inference server: arrakis.cs.rutgers.edu:{port}")
            self.log_manager.log_info(f"Environment configs: {env_configs}")
        
        return None
    
    def wait_for_evaluations(self, evaluation_ids: List[str], timeout: int = None):
        """Wait for all evaluations to complete"""
        if self.cfg.coordination != "ssh":
            self.log_manager.log_info("Manual coordination mode - skipping automatic wait")
            return
        
        self.log_manager.log_info(f"Waiting for {len(evaluation_ids)} evaluations to complete...")
        
        completed = set()
        
        while len(completed) < len(evaluation_ids):
            for eval_id in evaluation_ids:
                if eval_id in completed:
                    continue
                
                status = self.coordinator.check_process_status(eval_id)
                if status in ["completed", "failed"]:
                    completed.add(eval_id)
                    self.log_manager.log_info(f"Evaluation {eval_id} {status}")
                    
                    if status == "completed":
                        # Process completed successfully
                        success, _stdout, stderr = self.coordinator.wait_for_process(eval_id, timeout=10)
                        if success:
                            self.log_manager.log_info(f"Evaluation {eval_id} completed successfully")
                        else:
                            self.log_manager.log_error(f"Evaluation {eval_id} failed: {stderr}")
            
            time.sleep(10)  # Check every 10 seconds
        
        self.log_manager.log_info(f"All evaluations completed. {len(completed)}/{len(evaluation_ids)} finished.")
    
    def collect_and_aggregate_results(self, model_runs: List[str]) -> str:
        """Collect and aggregate results from all model runs"""
        self.log_manager.log_info("Collecting and aggregating results...")
        
        # Find results files for each model
        results_files = self.results_aggregator.find_results_files(model_runs)
        
        if not results_files:
            self.log_manager.log_error("No results files found")
            return ""
        
        # Aggregate results for each model
        all_model_stats = {}
        for model_name, files in results_files.items():
            self.log_manager.log_info(f"Aggregating results for {model_name}")
            stats = self.results_aggregator.aggregate_model_results(model_name, files)
            if stats:
                all_model_stats[model_name] = stats
        
        if not all_model_stats:
            self.log_manager.log_error("No valid results to aggregate")
            return ""
        
        # Create comparison report
        comparison_df = self.results_aggregator.create_comparison_report(all_model_stats)
        
        # Save aggregated results
        batch_dir = self.results_aggregator.save_aggregated_results(all_model_stats, comparison_df)
        
        # Generate summary report
        summary_report = self.results_aggregator.generate_summary_report(batch_dir)
        self.log_manager.log_info(f"Generated summary report:\n{summary_report}")
        
        return batch_dir
    
    def cleanup(self):
        """Clean up all resources and processes"""
        self.log_manager.log_info("Cleaning up resources...")
        
        # Clean up job queue manager first
        try:
            self.job_queue_manager.cleanup_all()
        except Exception as e:
            self.log_manager.log_error(f"Error cleaning up job queue manager: {e}")
        
        # Stop inference servers and collect their logs
        for server_id, server_info in self.inference_servers.items():
            try:
                process = server_info['process']
                if process.poll() is None:
                    self.log_manager.log_info(f"Terminating inference server {server_id}")
                    process.terminate()
                    try:
                        return_code = process.wait(timeout=10)
                        self.log_manager.log_process_end(server_id, return_code)
                    except subprocess.TimeoutExpired:
                        self.log_manager.log_warning(f"Force killing inference server {server_id}")
                        process.kill()
                        return_code = process.wait()
                        self.log_manager.log_process_end(server_id, return_code)
                else:
                    # Process already finished
                    return_code = process.returncode
                    self.log_manager.log_process_end(server_id, return_code)
                    
            except Exception as e:
                self.log_manager.log_error(f"Error stopping inference server {server_id}: {e}", exc_info=True)
        
        # Clean up SSH coordinator
        if hasattr(self.coordinator, 'cleanup_all_processes'):
            self.coordinator.cleanup_all_processes()
        
        # Release all resources
        self.resource_manager.cleanup_all()
        
        # Log final summary
        summary = self.log_manager.get_process_summary()
        self.log_manager.log_info(f"Process summary: {summary['total_processes']} total, "
                                 f"{summary['completed']} completed, {summary['failed']} failed")
        
        # Cleanup log manager (this closes files and writes final state)
        self.log_manager.cleanup()
        
        print(f"All logs saved to: {self.log_manager.experiment_dir}")  # Use print for final message
    
    def run_pipeline(self):
        """Run the complete evaluation pipeline"""
        try:
            self.log_manager.log_info("Starting Auto Evaluation Pipeline")
            
            # Step 1: Discover model runs
            model_runs = self.discover_model_runs()
            if not model_runs:
                self.log_manager.log_error("No model runs found to evaluate")
                return
            
            self.log_manager.log_info(f"Found {len(model_runs)} model runs to evaluate")
            
            # Step 2: Build job queue and start processing
            is_split = getattr(self.cfg, 'use_split_inference', False)
            obstacle_path = getattr(self.cfg, 'obstacle_run_path', None) if is_split else None
            goal_path = getattr(self.cfg, 'goal_run_path', None) if is_split else None
            
            # Build job queue
            self._build_job_queue(model_runs, is_split, obstacle_path, goal_path)
            
            if self.job_queue_manager.job_queue.qsize() == 0:
                self.log_manager.log_error("No jobs created for evaluation")
                return
            
            # Step 3: Start job queue processing
            self.log_manager.log_info(f"Starting job queue processing with {self.job_queue_manager.job_queue.qsize()} jobs")
            
            # Start processing jobs
            self.job_queue_manager.start_processing()
            
            # Wait for all jobs to complete
            self.job_queue_manager.wait_for_completion()
            
            # Step 4: Collect and aggregate results
            batch_dir = self.collect_and_aggregate_results(model_runs)
            if batch_dir:
                self.log_manager.log_info(f"Evaluation pipeline completed successfully!")
                self.log_manager.log_info(f"Results saved to: {batch_dir}")
            else:
                self.log_manager.log_error("Failed to aggregate results")
            
        except Exception as e:
            self.log_manager.log_error(f"Pipeline failed with error: {e}")
            raise
        finally:
            # Always cleanup
            self.cleanup()

@hydra.main(config_path="../config", config_name="auto_eval.yaml", version_base=None)
def main(cfg: DictConfig):
    logger.info("Auto Evaluation Pipeline Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Validate SSH connection if using SSH coordination
    if cfg.coordination == "ssh":
        coordinator = SSHCoordinator(cfg.ssh.host, cfg.ssh.ml4kp_path, cfg.ssh.eval_script_path)
        if not coordinator.test_ssh_connection():
            logger.error("SSH connection test failed")
            return
        if not coordinator.check_ml4kp_path():
            logger.error("ML4KP path check failed")
            return
    
    # Run pipeline
    pipeline = AutoEvaluationPipeline(cfg)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()