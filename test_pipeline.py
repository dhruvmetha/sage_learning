#!/usr/bin/env python
"""
Simple test script to validate pipeline components
"""
import sys
import logging
from pathlib import Path

# Add src to path so we can import utils
sys.path.append('src')

from utils.gpu_manager import ResourceManager
from utils.ssh_coordinator import SSHCoordinator
from utils.results_aggregator import ResultsAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_manager():
    """Test GPU and resource management"""
    logger.info("Testing GPU Manager...")
    
    rm = ResourceManager(available_gpus=[0, 1, 2], max_gpus_to_use=2)
    
    # Test GPU allocation
    resources1 = rm.allocate_resources("test_process_1")
    if resources1:
        gpu1, port1 = resources1
        logger.info(f"✅ Allocated GPU {gpu1}, Port {port1}")
    else:
        logger.error("❌ Failed to allocate resources")
        return False
    
    # Test port allocation
    resources2 = rm.allocate_resources("test_process_2")
    if resources2:
        gpu2, port2 = resources2
        logger.info(f"✅ Allocated GPU {gpu2}, Port {port2}")
    else:
        logger.error("❌ Failed to allocate second set of resources")
        return False
    
    # Test cleanup
    allocations = rm.get_all_allocations()
    logger.info(f"Active allocations: {len(allocations)}")
    
    rm.cleanup_all()
    allocations_after = rm.get_all_allocations()
    logger.info(f"Allocations after cleanup: {len(allocations_after)}")
    
    return len(allocations_after) == 0

def test_ssh_coordinator():
    """Test SSH coordinator"""
    logger.info("Testing SSH Coordinator...")
    
    coord = SSHCoordinator(
        ssh_host="westeros.cs.rutgers.edu",
        ml4kp_path="/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp",
        eval_script_path="executables/utils/eval_namo.py"
    )
    
    # Test SSH connection
    if coord.test_ssh_connection():
        logger.info("✅ SSH connection successful")
    else:
        logger.error("❌ SSH connection failed")
        return False
    
    # Test ML4KP path
    if coord.check_ml4kp_path():
        logger.info("✅ ML4KP path exists")
    else:
        logger.error("❌ ML4KP path not found")
        return False
    
    return True

def test_results_aggregator():
    """Test results aggregation"""
    logger.info("Testing Results Aggregator...")
    
    ra = ResultsAggregator("test_results")
    
    # Look for any existing results files
    sample_outputs = []
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        for date_dir in outputs_dir.iterdir():
            if date_dir.is_dir():
                for time_dir in date_dir.iterdir():
                    if time_dir.is_dir() and (time_dir / "checkpoints").exists():
                        sample_outputs.append(str(time_dir))
                        if len(sample_outputs) >= 2:
                            break
                if len(sample_outputs) >= 2:
                    break
    
    if sample_outputs:
        logger.info(f"Found sample outputs: {sample_outputs}")
        
        # Try to find results files
        results_files = ra.find_results_files(sample_outputs)
        logger.info(f"Found results files: {list(results_files.keys())}")
        
        if results_files:
            logger.info("✅ Results aggregator can find results files")
        else:
            logger.info("ℹ️  No existing results files found (this is OK)")
        
        return True
    else:
        logger.info("ℹ️  No sample output directories found (this is OK)")
        return True

def test_model_discovery():
    """Test model discovery logic"""
    logger.info("Testing Model Discovery...")
    
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        logger.info("ℹ️  No outputs directory found")
        return True
    
    # Find directories with completed training
    completed_runs = []
    
    date_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], 
                      key=lambda x: x.stat().st_mtime, reverse=True)
    
    for date_dir in date_dirs:
        time_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()],
                         key=lambda x: x.stat().st_mtime, reverse=True)
        
        for time_dir in time_dirs:
            checkpoint_dir = time_dir / "checkpoints"
            if checkpoint_dir.exists():
                epoch_checkpoints = list(checkpoint_dir.glob("epoch=*.ckpt"))
                if epoch_checkpoints:
                    completed_runs.append(str(time_dir))
                    logger.info(f"Found completed run: {time_dir}")
                    
                    if len(completed_runs) >= 3:
                        break
        
        if len(completed_runs) >= 3:
            break
    
    logger.info(f"✅ Discovered {len(completed_runs)} completed training runs")
    return True

def main():
    """Run all tests"""
    logger.info("🚀 Starting Pipeline Component Tests")
    
    tests = [
        ("GPU Manager", test_gpu_manager),
        ("SSH Coordinator", test_ssh_coordinator), 
        ("Results Aggregator", test_results_aggregator),
        ("Model Discovery", test_model_discovery)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Pipeline is ready to use.")
        logger.info("   Run: python src/auto_eval.py --help")
    else:
        logger.info("\n⚠️  Some tests failed. Check the logs above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)