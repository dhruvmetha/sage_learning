import subprocess
import psutil
import socket
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

class GPUManager:
    def __init__(self, available_gpus: List[int], max_gpus_to_use: int = 3):
        self.available_gpus = available_gpus
        self.max_gpus_to_use = max_gpus_to_use
        self.allocated_gpus = {}  # {gpu_id: process_info}
        
    def get_gpu_memory_usage(self, gpu_id: int) -> float:
        """Get GPU memory usage percentage for given GPU ID"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--id=' + str(gpu_id), 
                '--query-gpu=memory.used,memory.total', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(', '))
                return (used / total) * 100
            else:
                logger.warning(f"Could not get memory usage for GPU {gpu_id}")
                return 100.0  # Assume full if can't check
        except Exception as e:
            logger.error(f"Error checking GPU {gpu_id} memory: {e}")
            return 100.0
    
    def get_free_gpus(self, memory_threshold: float = 10.0) -> List[int]:
        """Get list of GPUs with memory usage below threshold"""
        free_gpus = []
        for gpu_id in self.available_gpus:
            if gpu_id not in self.allocated_gpus:
                memory_usage = self.get_gpu_memory_usage(gpu_id)
                if memory_usage < memory_threshold:
                    free_gpus.append(gpu_id)
        return free_gpus
    
    def allocate_gpu(self, process_name: str) -> Optional[int]:
        """Allocate a free GPU for a process"""
        if len(self.allocated_gpus) >= self.max_gpus_to_use:
            logger.warning(f"Maximum GPU allocation reached ({self.max_gpus_to_use})")
            return None
            
        free_gpus = self.get_free_gpus()
        if not free_gpus:
            logger.warning("No free GPUs available")
            return None
            
        gpu_id = free_gpus[0]
        self.allocated_gpus[gpu_id] = {
            'process_name': process_name,
            'allocated_at': psutil.time.time()
        }
        logger.info(f"Allocated GPU {gpu_id} to {process_name}")
        return gpu_id
    
    def release_gpu(self, gpu_id: int):
        """Release a GPU from allocation"""
        if gpu_id in self.allocated_gpus:
            process_name = self.allocated_gpus[gpu_id]['process_name']
            del self.allocated_gpus[gpu_id]
            logger.info(f"Released GPU {gpu_id} from {process_name}")
    
    def get_allocated_gpus(self) -> dict:
        """Get currently allocated GPUs"""
        return self.allocated_gpus.copy()

class PortManager:
    def __init__(self, start_port: int = 5556, end_port: int = 5570):
        self.start_port = start_port
        self.end_port = end_port
        self.allocated_ports = set()
    
    def is_port_free(self, port: int) -> bool:
        """Check if a port is free"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                result = sock.connect_ex(('localhost', port))
                return result != 0  # Port is free if connection fails
        except Exception:
            return False
    
    def get_free_port(self) -> Optional[int]:
        """Get next available free port"""
        for port in range(self.start_port, self.end_port + 1):
            if port not in self.allocated_ports and self.is_port_free(port):
                self.allocated_ports.add(port)
                logger.info(f"Allocated port {port}")
                return port
        logger.warning("No free ports available")
        return None
    
    def release_port(self, port: int):
        """Release a port from allocation"""
        if port in self.allocated_ports:
            self.allocated_ports.remove(port)
            logger.info(f"Released port {port}")
    
    def get_allocated_ports(self) -> set:
        """Get currently allocated ports"""
        return self.allocated_ports.copy()

class ResourceManager:
    """Combined GPU and Port management"""
    def __init__(self, available_gpus: List[int], max_gpus_to_use: int = 3, 
                 start_port: int = 5556, end_port: int = 5570):
        self.gpu_manager = GPUManager(available_gpus, max_gpus_to_use)
        self.port_manager = PortManager(start_port, end_port)
        self.allocations = {}  # {allocation_id: {gpu_id, port, process_name}}
    
    def allocate_resources(self, process_name: str) -> Optional[Tuple[int, int]]:
        """Allocate both GPU and port for a process"""
        gpu_id = self.gpu_manager.allocate_gpu(process_name)
        if gpu_id is None:
            return None
            
        port = self.port_manager.get_free_port()
        if port is None:
            # Release GPU if port allocation failed
            self.gpu_manager.release_gpu(gpu_id)
            return None
        
        allocation_id = f"{process_name}_{gpu_id}_{port}"
        self.allocations[allocation_id] = {
            'gpu_id': gpu_id,
            'port': port,
            'process_name': process_name
        }
        
        logger.info(f"Allocated resources: GPU {gpu_id}, Port {port} for {process_name}")
        return gpu_id, port
    
    def release_resources(self, allocation_id: str):
        """Release both GPU and port for an allocation"""
        if allocation_id in self.allocations:
            allocation = self.allocations[allocation_id]
            self.gpu_manager.release_gpu(allocation['gpu_id'])
            self.port_manager.release_port(allocation['port'])
            del self.allocations[allocation_id]
            logger.info(f"Released resources for {allocation['process_name']}")
    
    def get_all_allocations(self) -> dict:
        """Get all current resource allocations"""
        return self.allocations.copy()
    
    def cleanup_all(self):
        """Release all allocated resources"""
        for allocation_id in list(self.allocations.keys()):
            self.release_resources(allocation_id)