#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality Checker for Keyframe Evaluation Results
Analyzes eval_results.csv and provides quality rating
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import json
import os

class KeyframeQualityChecker:
    def __init__(self):
        # Định nghĩa ngưỡng cho từng metric dựa trên ý nghĩa thực tế
        self.thresholds = {
            # I. Tính Đại diện (Representativeness)
            # RecErr (thấp = tốt, gần 0)
            'RecErr': {'excellent': 0.3, 'good': 0.5, 'fair': 0.8, 'poor': float('inf')},
            
            # Frechet Distance (thấp = tốt, gần 0)
            'Frechet': {'excellent': 0.8, 'good': 1.2, 'fair': 1.6, 'poor': float('inf')},
            
            # II. Tính Bao phủ (Coverage)
            # Scene Coverage (cao = tốt, lý tưởng = 1.0)
            'SceneCoverage': {'excellent': 0.95, 'good': 0.85, 'fair': 0.70, 'poor': 0.0},
            
            # Temporal Coverage (cao = tốt, lý tưởng = 1.0)
            'TemporalCoverage@tau': {'excellent': 0.7, 'good': 0.5, 'fair': 0.3, 'poor': 0.0},
            
            # III. Tính Đa dạng (Diversity)
            # Redundancy Mean Cosine (thấp = tốt, gần 0 = ít redundant)
            'RedundancyMeanCos': {'excellent': 0.2, 'good': 0.4, 'fair': 0.6, 'poor': float('inf')},
            
            # Min Pairwise Distance (cao = tốt, gần 1.0 = đa dạng)
            'MinPairwiseDist': {'excellent': 0.6, 'good': 0.4, 'fair': 0.2, 'poor': 0.0},
            
            # IV. Chất lượng Hình ảnh (Quality Proxies)
            # Sharpness_med (cao = tốt, không mờ)
            'Sharpness_med': {'excellent': 200, 'good': 120, 'fair': 80, 'poor': 0},
            
            # Exposure_med (gần 128 ± 40 = tốt, tránh quá sáng/tối)
            'Exposure_med': {'excellent': (88, 168), 'good': (60, 196), 'fair': (40, 216), 'poor': (0, 255)},
            
            # Noise_med (thấp = tốt, gần 0)
            'Noise_med': {'excellent': 3.0, 'good': 5.0, 'fair': 7.0, 'poor': float('inf')},
        }
        
        # Nhóm metrics theo ý nghĩa
        self.metric_groups = {
            'Representativeness': ['RecErr', 'Frechet'],
            'Coverage': ['SceneCoverage', 'TemporalCoverage@tau'],
            'Diversity': ['RedundancyMeanCos', 'MinPairwiseDist'],
            'Image Quality': ['Sharpness_med', 'Exposure_med', 'Noise_med']
        }
        
        # Trọng số cho từng nhóm
        self.group_weights = {
            'Representativeness': 0.3,
            'Coverage': 0.3,
            'Diversity': 0.2,
            'Image Quality': 0.2
        }
        
        # Trọng số cho từng metric trong nhóm của nó
        self.metric_weights = {
            'RecErr': 0.5,
            'Frechet': 0.5,
            'SceneCoverage': 0.6,
            'TemporalCoverage@tau': 0.4,
            'RedundancyMeanCos': 0.5,
            'MinPairwiseDist': 0.5,
            'Sharpness_med': 0.4,
            'Exposure_med': 0.3,
            'Noise_med': 0.3,
        }

    def rate_metric(self, metric_name: str, value: float) -> Tuple[str, int]:
        """Rate a single metric and return (rating, score)"""
        if metric_name not in self.thresholds:
            return 'unknown', 0
            
        thresholds = self.thresholds[metric_name]
        
        # Special handling for Exposure (range-based)
        if metric_name == 'Exposure_med':
            if thresholds['excellent'][0] <= value <= thresholds['excellent'][1]:
                return 'excellent', 4
            elif thresholds['good'][0] <= value <= thresholds['good'][1]:
                return 'good', 3
            elif thresholds['fair'][0] <= value <= thresholds['fair'][1]:
                return 'fair', 2
            else:
                return 'poor', 1
        
        # For "higher is better" metrics
        if metric_name in ['SceneCoverage', 'TemporalCoverage@tau', 'MinPairwiseDist', 'Sharpness_med']:
            if value >= thresholds['excellent']:
                return 'excellent', 4
            elif value >= thresholds['good']:
                return 'good', 3
            elif value >= thresholds['fair']:
                return 'fair', 2
            else:
                return 'poor', 1
        
        # For "lower is better" metrics
        else:
            if value <= thresholds['excellent']:
                return 'excellent', 4
            elif value <= thresholds['good']:
                return 'good', 3
            elif value <= thresholds['fair']:
                return 'fair', 2
            else:
                return 'poor', 1

    def analyze_results(self, csv_path: str) -> Dict:
        """Analyze eval_results.csv and return comprehensive quality assessment"""
        df = pd.read_csv(csv_path)
        metrics = dict(zip(df['metric'], df['value']))
        
        results = {
            'file_path': csv_path,
            'metrics': {},
            'group_scores': {},
            'overall_score': 0,
            'overall_rating': '',
            'recommendations': [],
            'summary': {}
        }
        
        # Analyze each metric
        for metric, value in metrics.items():
            if metric in self.thresholds:
                rating, score = self.rate_metric(metric, value)
                
                results['metrics'][metric] = {
                    'value': value,
                    'rating': rating,
                    'score': score,
                }
        
        # Calculate group scores
        group_weighted_scores = {}
        for group_name, group_metrics in self.metric_groups.items():
            group_score = 0
            group_weight_sum = 0
            
            for metric in group_metrics:
                if metric in results['metrics']:
                    metric_data = results['metrics'][metric]
                    metric_weight = self.metric_weights.get(metric, 1.0)
                    group_score += metric_data['score'] * metric_weight
                    group_weight_sum += metric_weight
            
            if group_weight_sum > 0:
                avg_group_score = group_score / group_weight_sum
                group_weighted_scores[group_name] = avg_group_score
                
                results['group_scores'][group_name] = {
                    'score': round(avg_group_score, 3),
                    'rating': self._score_to_rating(avg_group_score)
                }
        
        # Calculate overall score (weighted by group weights)
        total_weighted_score = 0
        total_weights = 0
        for group_name, group_score in group_weighted_scores.items():
            group_weight = self.group_weights.get(group_name, 0)
            total_weighted_score += group_score * group_weight
            total_weights += group_weight
        
        if total_weights > 0:
            overall_score = total_weighted_score / total_weights
            results['overall_score'] = round(overall_score, 3)
            results['overall_rating'] = self._score_to_rating(overall_score)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results['metrics'])
        
        # Summary statistics
        results['summary'] = {
            'num_keyframes': int(metrics.get('NumKeys', 0)),
            'num_embeddings': int(metrics.get('NumAllEmbed', 0)),
            'keyframe_ratio': round(metrics.get('NumKeys', 0) / max(metrics.get('NumAllEmbed', 1), 1), 3),
            'critical_issues': self._find_critical_issues(results['metrics'])
        }
        
        return results
    
    def _score_to_rating(self, score: float) -> str:
        """Convert numeric score to rating"""
        if score >= 3.5:
            return 'excellent'
        elif score >= 2.5:
            return 'good'
        elif score >= 1.5:
            return 'fair'
        else:
            return 'poor'

    def batch_analyze_outputs_eval(self, outputs_eval_dir: str = "outputs_eval") -> Dict:
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

"""
USAGE EXAMPLES:

# Analyze all videos in outputs_eval directory
python quality_checker.py

# Specify custom directory and output files
python quality_checker.py --eval_dir outputs_eval --output_report my_report.txt

# Quick usage in Python
from quality_checker import KeyframeQualityChecker
checker = KeyframeQualityChecker()
results = checker.batch_analyze_outputs_eval("outputs_eval")
checker.generate_comprehensive_report(results, "report.txt")
"""