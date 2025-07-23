import pandas as pd
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime
import csv

logger = logging.getLogger(__name__)

class ResultsAggregator:
    def __init__(self, results_dir: str = "eval_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def find_results_files(self, output_dirs: List[str]) -> Dict[str, List[Path]]:
        """Find all results.txt files in the given output directories"""
        results_files = {}
        
        for output_dir in output_dirs:
            output_path = Path(output_dir)
            model_name = self._extract_model_name(output_path)
            
            # Look for results files in various subdirectories
            result_patterns = [
                "ood_results_25/results.txt",
                "ood_results_hard/results.txt", 
                "ood_results_medium/results.txt",
                "results/results.txt",
                "results_hard/results.txt",
                "results_medium/results.txt"
            ]
            
            model_results = []
            for pattern in result_patterns:
                results_file = output_path / pattern
                if results_file.exists():
                    model_results.append(results_file)
            
            if model_results:
                results_files[model_name] = model_results
            else:
                logger.warning(f"No results files found for model {model_name} in {output_dir}")
        
        return results_files
    
    def _extract_model_name(self, output_path: Path) -> str:
        """Extract a meaningful model name from output path"""
        # Try to extract from hydra config
        hydra_config = output_path / ".hydra" / "config.yaml"
        if hydra_config.exists():
            try:
                import yaml
                with open(hydra_config, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Extract model type from config
                model_type = config.get('model', {}).get('_target_', 'unknown')
                if 'dit' in model_type.lower():
                    model_name = 'dit'
                elif 'unet' in model_type.lower():
                    model_name = 'unet'
                else:
                    model_name = 'unknown'
                
                # Add timestamp for uniqueness
                timestamp = output_path.name
                return f"{model_name}_{timestamp}"
                
            except Exception as e:
                logger.warning(f"Failed to parse hydra config: {e}")
        
        # Fallback to directory name
        return output_path.name
    
    def parse_results_file(self, results_file: Path) -> pd.DataFrame:
        """Parse a single results.txt file into DataFrame"""
        try:
            # Results format: config_name,success,action_steps,decision_idx,reachable_selection,reachable_checks
            df = pd.read_csv(results_file, names=[
                'config_name', 'success', 'action_steps', 'decision_idx', 
                'reachable_selection', 'reachable_checks'
            ])
            
            # Add metadata
            df['results_file'] = str(results_file)
            df['env_set'] = self._infer_env_set_from_path(results_file)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse results file {results_file}: {e}")
            return pd.DataFrame()
    
    def _infer_env_set_from_path(self, results_file: Path) -> str:
        """Infer environment set name from file path"""
        path_str = str(results_file)
        
        if 'ood_results' in path_str:
            if 'hard' in path_str:
                return 'ood_hard'
            elif 'medium' in path_str:
                return 'ood_medium'
            else:
                return 'ood'
        elif 'hard' in path_str:
            return 'hard'
        elif 'medium' in path_str:
            return 'medium'
        else:
            return 'standard'
    
    def aggregate_model_results(self, model_name: str, results_files: List[Path]) -> Dict:
        """Aggregate results for a single model across all environment sets"""
        all_results = []
        
        for results_file in results_files:
            df = self.parse_results_file(results_file)
            if not df.empty:
                all_results.append(df)
        
        if not all_results:
            logger.warning(f"No valid results found for model {model_name}")
            return {}
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Calculate aggregate statistics
        stats = {}
        
        # Overall statistics
        stats['overall'] = self._calculate_stats(combined_df)
        
        # Per-environment set statistics
        stats['by_env_set'] = {}
        for env_set in combined_df['env_set'].unique():
            env_df = combined_df[combined_df['env_set'] == env_set]
            stats['by_env_set'][env_set] = self._calculate_stats(env_df)
        
        # Per-environment statistics
        stats['by_environment'] = {}
        for config_name in combined_df['config_name'].unique():
            env_df = combined_df[combined_df['config_name'] == config_name]
            stats['by_environment'][config_name] = self._calculate_stats(env_df)
        
        return stats
    
    def _calculate_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics for a DataFrame of results"""
        if df.empty:
            return {}
        
        # Success rate
        success_rate = df['success'].mean() if 'success' in df.columns else 0
        
        # Action steps statistics (only for successful trials)
        successful_trials = df[df['success'] == 1] if 'success' in df.columns else df
        
        stats = {
            'total_trials': len(df),
            'successful_trials': len(successful_trials),
            'success_rate': success_rate,
        }
        
        if len(successful_trials) > 0 and 'action_steps' in successful_trials.columns:
            stats.update({
                'avg_action_steps': successful_trials['action_steps'].mean(),
                'median_action_steps': successful_trials['action_steps'].median(), 
                'std_action_steps': successful_trials['action_steps'].std(),
                'min_action_steps': successful_trials['action_steps'].min(),
                'max_action_steps': successful_trials['action_steps'].max(),
            })
        
        # Decision statistics
        if 'decision_idx' in df.columns:
            stats.update({
                'avg_decisions': df['decision_idx'].mean(),
                'max_decisions': df['decision_idx'].max(),
            })
        
        # Reachability statistics
        if 'reachable_selection' in df.columns and 'reachable_checks' in df.columns:
            total_checks = df['reachable_checks'].sum()
            total_selections = df['reachable_selection'].sum()
            stats.update({
                'total_reachable_checks': total_checks,
                'total_reachable_selections': total_selections,
                'reachable_selection_rate': total_selections / total_checks if total_checks > 0 else 0
            })
        
        return stats
    
    def create_comparison_report(self, all_model_stats: Dict[str, Dict]) -> pd.DataFrame:
        """Create comparison report across all models"""
        comparison_data = []
        
        for model_name, stats in all_model_stats.items():
            if 'overall' not in stats:
                continue
                
            overall_stats = stats['overall']
            
            row = {
                'model_name': model_name,
                'total_trials': overall_stats.get('total_trials', 0),
                'success_rate': overall_stats.get('success_rate', 0),
                'avg_action_steps': overall_stats.get('avg_action_steps', 0),
                'median_action_steps': overall_stats.get('median_action_steps', 0),
                'reachable_selection_rate': overall_stats.get('reachable_selection_rate', 0),
            }
            
            # Add per-environment set success rates
            for env_set, env_stats in stats.get('by_env_set', {}).items():
                row[f'{env_set}_success_rate'] = env_stats.get('success_rate', 0)
                row[f'{env_set}_trials'] = env_stats.get('total_trials', 0)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def save_aggregated_results(self, all_model_stats: Dict[str, Dict], 
                               comparison_df: pd.DataFrame, 
                               batch_name: Optional[str] = None) -> str:
        """Save all aggregated results to files"""
        
        if batch_name is None:
            batch_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_evaluation_batch")
        
        batch_dir = self.results_dir / batch_name
        batch_dir.mkdir(exist_ok=True)
        
        # Save overall summary
        summary_file = batch_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_model_stats, f, indent=2, default=str)
        
        # Save comparison report
        comparison_file = batch_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        # Save detailed stats per model
        for model_name, stats in all_model_stats.items():
            model_dir = batch_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Save aggregated stats
            stats_file = model_dir / "aggregated_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            # Save per-environment results
            if 'by_environment' in stats:
                env_results = []
                for env_name, env_stats in stats['by_environment'].items():
                    env_stats['environment'] = env_name
                    env_results.append(env_stats)
                
                env_df = pd.DataFrame(env_results)
                env_file = model_dir / "per_environment_results.csv"
                env_df.to_csv(env_file, index=False)
        
        logger.info(f"Saved aggregated results to {batch_dir}")
        return str(batch_dir)
    
    def generate_summary_report(self, batch_dir: str) -> str:
        """Generate a human-readable summary report"""
        batch_path = Path(batch_dir)
        
        # Load comparison data
        comparison_file = batch_path / "model_comparison.csv"
        if not comparison_file.exists():
            logger.error("Comparison file not found")
            return ""
        
        comparison_df = pd.read_csv(comparison_file)
        
        # Generate report
        report_lines = []
        report_lines.append("# Evaluation Results Summary")
        report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Results directory: {batch_dir}")
        report_lines.append("")
        
        # Overall comparison
        report_lines.append("## Model Comparison")
        report_lines.append("")
        
        # Sort by success rate
        sorted_df = comparison_df.sort_values('success_rate', ascending=False)
        
        for _, row in sorted_df.iterrows():
            model_name = row['model_name']
            success_rate = row['success_rate'] * 100
            total_trials = row['total_trials']
            avg_steps = row.get('avg_action_steps', 0)
            
            report_lines.append(f"**{model_name}**")
            report_lines.append(f"- Success Rate: {success_rate:.1f}% ({total_trials} trials)")
            if avg_steps > 0:
                report_lines.append(f"- Average Action Steps: {avg_steps:.1f}")
            report_lines.append("")
        
        # Best performing model
        if not sorted_df.empty:
            best_model = sorted_df.iloc[0]
            report_lines.append(f"## Best Performing Model: {best_model['model_name']}")
            report_lines.append(f"Success Rate: {best_model['success_rate']*100:.1f}%")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_file = batch_path / "summary_report.md"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        return report_text