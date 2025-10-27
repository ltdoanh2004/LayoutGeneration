#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Results Analyzer
Analyzes eval_results.json and aggregates metrics from evaluator.py
Strictly follows evaluator.py metric definitions without thresholds or ratings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
import os

class EvaluationResultsAnalyzer:
    """
    Analyzes evaluation results directly from evaluator.py output.
    No thresholds or ratings - just raw metric values.
    """
    
    def __init__(self):
        # Metrics from evaluator.py eval_one_set() function
        # Grouped by category for reference only (no scoring/weighting)
        self.metric_groups = {
            'Representativeness': ['RecErr', 'Frechet'],
            'Coverage': ['SceneCoverage', 'TemporalCoverage@tau'],
            'Diversity': ['RedundancyMeanCos', 'MinPairwiseDist'],
            'Image_Quality': ['Sharpness_med', 'Exposure_med', 'Noise_med'],
            'Metadata': ['NumKeys', 'NumAllEmbed']
        }
        
        # All metrics from evaluator.py
        self.all_metrics = [
            'RecErr', 'Frechet', 'SceneCoverage', 'TemporalCoverage@tau',
            'RedundancyMeanCos', 'MinPairwiseDist', 'Sharpness_med', 'Exposure_med', 'Noise_med',
            'NumKeys', 'NumAllEmbed'
        ]

    def analyze_results(self, json_path: str) -> Dict:
        """
        Analyze eval_results.json directly from evaluator.py.
        Returns raw metric values without thresholds or ratings.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        except Exception as e:
            return {'error': f'Failed to load {json_path}: {e}'}
        
        # Extract all metrics
        result = {
            'file_path': json_path,
            'metrics': {}
        }
        
        for metric in self.all_metrics:
            if metric in metrics:
                result['metrics'][metric] = float(metrics[metric])
        
        return result

    def batch_analyze_outputs(self, outputs_dir: str = "outputs") -> Dict:
        """
        Automatically scan outputs_eval directory and analyze all eval_results.csv files
        """
        eval_path = Path(outputs_eval_dir)
        results = {}
        
        if not eval_path.exists():
            print(f"Directory {outputs_eval_dir} does not exist!")
            return {}
        
        print(f"Scanning directory: {eval_path}")
        
        # Find all eval_* subdirectories
        eval_dirs = [d for d in eval_path.iterdir() if d.is_dir() and d.name.startswith('eval_')]
        
        print(f"Found {len(eval_dirs)} evaluation directories")
        
        for eval_dir in sorted(eval_dirs):
            video_id = eval_dir.name.replace('eval_', '')
            csv_file = eval_dir / "eval_results.csv"
            
            if csv_file.exists():
                try:
                    print(f"Analyzing {video_id}...")
                    analysis = self.analyze_results(str(csv_file))
                    results[video_id] = analysis
                    print(f"✅ {video_id}: {analysis['overall_rating']} ({analysis['overall_score']}/4.0)")
                except Exception as e:
                    print(f"❌ Error analyzing {video_id}: {e}")
                    results[video_id] = {'error': str(e)}
            else:
                print(f"⚠️  No eval_results.csv found in {eval_dir}")
                results[video_id] = {'error': 'eval_results.csv not found'}
        
        return results

    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate improvement recommendations based on metric scores"""
        recommendations = []
        
        for metric, data in metrics.items():
            if data['score'] <= 2:  # Poor or Fair
                if metric == 'RecErr':
                    recommendations.append("High reconstruction error - consider more representative keyframes")
                elif metric == 'Frechet':
                    recommendations.append("High Frechet distance - keyframes may not capture content diversity well")
                elif metric == 'TemporalCoverage@tau':
                    recommendations.append("Low temporal coverage - increase number of keyframes or improve selection")
                elif metric == 'RedundancyMeanCos':
                    recommendations.append("High redundancy - remove similar keyframes")
                elif metric == 'MinPairwiseDist':
                    recommendations.append("Low diversity - select more diverse keyframes")
                elif metric == 'Sharpness_med':
                    recommendations.append("Low sharpness - avoid blurry frames")
                elif metric == 'Noise_med':
                    recommendations.append("High noise - apply noise reduction or select cleaner frames")
        
        return recommendations

    def _find_critical_issues(self, metrics: Dict) -> List[str]:
        """Identify critical quality issues"""
        critical = []
        
        for metric, data in metrics.items():
            if data['score'] == 1:  # Poor rating
                critical.append(f"{metric}: {data['rating']} ({data['value']:.4f})")
        
        return critical

    def generate_comprehensive_report(self, results: Dict, output_path: str = None) -> str:
        """Generate a comprehensive quality report for all videos"""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        error_results = {k: v for k, v in results.items() if 'error' in v}
        
        if not valid_results:
            return "No valid results to analyze."
        
        # Calculate statistics
        scores = [r['overall_score'] for r in valid_results.values()]
        ratings = [r['overall_rating'] for r in valid_results.values()]
        
        rating_counts = {r: ratings.count(r) for r in ['excellent', 'good', 'fair', 'poor']}
        
        # Calculate group statistics
        group_stats = {}
        for group_name in self.metric_groups.keys():
            group_scores = []
            for video_id, result in valid_results.items():
                if group_name in result['group_scores']:
                    group_scores.append(result['group_scores'][group_name]['score'])
            
            if group_scores:
                group_stats[group_name] = {
                    'avg': np.mean(group_scores),
                    'std': np.std(group_scores),
                    'min': min(group_scores),
                    'max': max(group_scores)
                }
        
        # Calculate metric statistics
        metric_stats = {}
        for video_id, result in valid_results.items():
            for metric, data in result['metrics'].items():
                if metric not in metric_stats:
                    metric_stats[metric] = []
                metric_stats[metric].append(data['value'])
        
        report = f"""
{'='*80}
COMPREHENSIVE KEYFRAME QUALITY REPORT (LPIPS-based Evaluation)
{'='*80}

📊 OVERALL STATISTICS:
{'-'*80}
- Total videos analyzed: {len(valid_results)}
- Videos with errors: {len(error_results)}
- Average quality score: {np.mean(scores):.3f}/4.0
- Standard deviation: {np.std(scores):.3f}
- Best score: {max(scores):.3f}
- Worst score: {min(scores):.3f}

📈 RATING DISTRIBUTION:
{'-'*80}
- Excellent (3.5-4.0): {rating_counts['excellent']:3d} videos ({rating_counts['excellent']/max(len(valid_results), 1)*100:5.1f}%)
- Good (2.5-3.4):     {rating_counts['good']:3d} videos ({rating_counts['good']/max(len(valid_results), 1)*100:5.1f}%)
- Fair (1.5-2.4):     {rating_counts['fair']:3d} videos ({rating_counts['fair']/max(len(valid_results), 1)*100:5.1f}%)
- Poor (1.0-1.4):     {rating_counts['poor']:3d} videos ({rating_counts['poor']/max(len(valid_results), 1)*100:5.1f}%)

🎯 GROUP SCORES (Mean ± Std):
{'-'*80}
"""
        
        for group_name in ['Representativeness', 'Coverage', 'Diversity', 'Image Quality']:
            if group_name in group_stats:
                stats = group_stats[group_name]
                rating = self._score_to_rating(stats['avg'])
                report += f"I. {group_name:25s}: {stats['avg']:.3f} ± {stats['std']:.3f} ({rating:10s}) [min: {stats['min']:.3f}, max: {stats['max']:.3f}]\n"
        
        report += f"\n📋 DETAILED METRIC STATISTICS:\n"
        report += f"{'-'*80}\n"
        
        for metric in ['RecErr', 'Frechet', 'SceneCoverage', 'TemporalCoverage@tau', 
                       'RedundancyMeanCos', 'MinPairwiseDist', 'Sharpness_med', 'Exposure_med', 'Noise_med']:
            if metric in metric_stats:
                values = metric_stats[metric]
                report += f"{metric:25s}: avg={np.mean(values):8.4f}, std={np.std(values):8.4f}, min={min(values):8.4f}, max={max(values):8.4f}\n"
        
        # Sort videos by score
        sorted_videos = sorted(valid_results.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        
        report += f"\n🏆 TOP 10 BEST VIDEOS:\n"
        report += f"{'-'*80}\n"
        for i, (video_id, analysis) in enumerate(sorted_videos[:10], 1):
            keyframes = analysis['summary']['num_keyframes']
            ratio = analysis['summary']['keyframe_ratio']
            report += f"{i:2d}. Video {video_id:8s}: Score={analysis['overall_score']:.3f} ({analysis['overall_rating']:10s}) | {keyframes:3d} keyframes (ratio: {ratio:.3f})\n"
        
        report += f"\n⚠️  TOP 10 WORST VIDEOS:\n"
        report += f"{'-'*80}\n"
        for i, (video_id, analysis) in enumerate(reversed(sorted_videos[-10:]), 1):
            keyframes = analysis['summary']['num_keyframes']
            ratio = analysis['summary']['keyframe_ratio']
            critical_count = len(analysis['summary']['critical_issues'])
            report += f"{i:2d}. Video {video_id:8s}: Score={analysis['overall_score']:.3f} ({analysis['overall_rating']:10s}) | {keyframes:3d} keyframes, {critical_count} critical issues\n"
        
        # Videos with critical issues
        critical_videos = [(vid, res) for vid, res in valid_results.items() if res['summary']['critical_issues']]
        if critical_videos:
            report += f"\n🔴 VIDEOS WITH CRITICAL ISSUES ({len(critical_videos)} videos):\n"
            report += f"{'-'*80}\n"
            for video_id, analysis in sorted(critical_videos, key=lambda x: len(x[1]['summary']['critical_issues']), reverse=True)[:15]:
                issues = analysis['summary']['critical_issues']
                issue_str = ' | '.join(issues[:2]) + ('...' if len(issues) > 2 else '')
                report += f"- Video {video_id:8s}: {len(issues)} issues - {issue_str}\n"
        
        if error_results:
            report += f"\n❌ ERRORS ({len(error_results)} videos):\n"
            report += f"{'-'*80}\n"
            for video_id, error_info in error_results.items():
                report += f"- Video {video_id}: {error_info['error']}\n"
        
        report += f"\n{'='*80}\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        
        return report

    def save_detailed_json(self, results: Dict, output_path: str):
        """Save detailed results as JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {output_path}")

    def create_summary_csv(self, results: Dict, output_path: str):
        """Create a summary CSV with key metrics for each video"""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        summary_data = []
        for video_id, analysis in valid_results.items():
            row = {
                'video_id': video_id,
                'overall_score': analysis['overall_score'],
                'overall_rating': analysis['overall_rating'],
                'num_keyframes': analysis['summary']['num_keyframes'],
                'keyframe_ratio': analysis['summary']['keyframe_ratio'],
                'critical_issues_count': len(analysis['summary']['critical_issues'])
            }
            
            # Add group scores
            for group_name, group_data in analysis['group_scores'].items():
                row[f'{group_name}_score'] = group_data['score']
                row[f'{group_name}_rating'] = group_data['rating']
            
            # Add individual metric values and ratings
            for metric, data in analysis['metrics'].items():
                row[f'{metric}_value'] = data['value']
                row[f'{metric}_rating'] = data['rating']
                row[f'{metric}_score'] = data['score']
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('overall_score', ascending=False)
        df.to_csv(output_path, index=False)
        print(f"Summary CSV saved to: {output_path}")


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze keyframe quality from outputs_eval directory")
    parser.add_argument("--eval_dir", default=r"/home/serverai/ltdoanh/LayoutGeneration/outputs_eval", help="Path to outputs_eval directory")
    parser.add_argument("--output_report", default="quality_report.txt", help="Output report file")
    parser.add_argument("--output_json", default="detailed_results.json", help="Detailed JSON output")
    parser.add_argument("--output_csv", default="summary_results.csv", help="Summary CSV output")
    
    args = parser.parse_args()
    
    print("🔍 Starting Keyframe Quality Analysis...")
    
    checker = KeyframeQualityChecker()
    
    # Batch analyze all videos in outputs_eval
    results = checker.batch_analyze_outputs_eval(args.eval_dir)
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"\n📊 Analysis complete! Processed {len(results)} videos")
    
    # Generate comprehensive report
    report = checker.generate_comprehensive_report(results, args.output_report)
    print(report)
    
    # Save detailed JSON
    checker.save_detailed_json(results, args.output_json)
    
    # Create summary CSV
    checker.create_summary_csv(results, args.output_csv)
    
    print(f"\n✅ All outputs saved:")
    print(f"   📄 Report: {args.output_report}")
    print(f"   📋 JSON: {args.output_json}")
    print(f"   📊 CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
