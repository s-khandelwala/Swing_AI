"""
Golf Swing Improvement System

Provides actionable feedback to improve golf swings based on quality scores.
Uses golf instruction principles (Golf Digest, MyTPI, Hackmotion) to suggest
specific improvements.

No RL needed - uses rule-based recommendations with optional learning from feedback.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class SwingImprovementAdvisor:
    """
    Provides swing improvement recommendations based on quality scores.
    
    Approach:
    1. Analyze quality scores to identify weaknesses
    2. Apply golf instruction principles to suggest improvements
    3. Prioritize suggestions (most impactful first)
    4. Optionally learn from feedback which suggestions work best
    """
    
    def __init__(self, learn_from_feedback=False, conservative_mode=True):
        """
        Args:
            learn_from_feedback: If True, tracks which suggestions are effective
            conservative_mode: If True, prioritizes minor optimizations over major changes
        """
        self.learn_from_feedback = learn_from_feedback
        self.conservative_mode = conservative_mode
        
        # Track suggestion effectiveness (if learning enabled)
        self.suggestion_history = []  # List of (suggestion, score_before, score_after)
        
        # Golf instruction principles (thresholds for good vs needs improvement)
        self.thresholds = {
            'setup_quality': 0.75,
            'tempo_rhythm': 0.70,
            'weight_shift': 0.65,
            'body_rotation': 0.70,
            'impact_quality': 0.70,
            'followthrough': 0.75,
            'balance': 0.80,
            'consistency': 0.70
        }
        
        # Critical threshold: only suggest major changes if score is below this
        self.critical_threshold = 0.40  # Very low score = critical issue
        self.major_change_gap_threshold = 0.35  # Large gap = may need major change
        
        # Improvement suggestions database
        self.suggestions_db = self._load_suggestions_database()
    
    def _load_suggestions_database(self) -> Dict:
        """
        Load golf instruction-based improvement suggestions.
        Based on Golf Digest, MyTPI, Hackmotion principles.
        """
        return {
            'setup_quality': {
                'low': [
                    {
                        'priority': 'high',
                        'impact_level': 'moderate',  # Moderate: affects setup but not full swing mechanics
                        'area': 'Stance & Posture',
                        'issue': 'Unstable address position',
                        'suggestion': 'Focus on establishing a stable, athletic stance. Keep weight balanced, knees slightly bent, and spine angled forward (S-Posture). Hold this position for 2-3 seconds before starting your swing.',
                        'drill': 'Practice address position: Set up, hold for 5 seconds without moving, then step away. Repeat 10 times.',
                        'key_points': ['Balanced weight distribution', 'Knees slightly bent', 'Spine angled forward', 'Hold position']
                    },
                    {
                        'priority': 'medium',
                        'impact_level': 'minor',  # Minor: small alignment tweak
                        'area': 'Alignment',
                        'issue': 'Inconsistent setup alignment',
                        'suggestion': 'Use alignment sticks or clubs on the ground to ensure consistent body and clubface alignment to target.',
                        'drill': 'Place alignment stick on ground, practice setting up parallel to it.',
                        'key_points': ['Body parallel to target line', 'Clubface square to target']
                    }
                ],
                'optimization': [  # Minor optimizations that don't change mechanics
                    {
                        'priority': 'low',
                        'impact_level': 'minor',
                        'area': 'Setup Consistency',
                        'issue': 'Minor setup inconsistencies',
                        'suggestion': 'Fine-tune your setup routine. Ensure you take the same amount of time, same pre-shot routine, and same positioning every time. Small consistency improvements can lead to better results.',
                        'drill': 'Practice your pre-shot routine 10 times: Same steps, same timing, same positioning.',
                        'key_points': ['Consistent routine', 'Same timing', 'Same positioning']
                    }
                ]
            },
            'tempo_rhythm': {
                'low': [
                    {
                        'priority': 'high',
                        'impact_level': 'moderate',  # Moderate: affects timing but not mechanics
                        'area': 'Tempo',
                        'issue': 'Rushed or inconsistent tempo',
                        'suggestion': 'Focus on smooth, unhurried transition. The backswing should set up the downswing - don\'t rush the transition. Think "smooth and steady" rather than "fast and powerful".',
                        'drill': 'Practice with a metronome: Backswing on "1-2", transition on "3", downswing on "4-5". Start slow, gradually increase speed while maintaining rhythm.',
                        'key_points': ['Smooth transition', 'Backswing sets up downswing', 'Consistent rhythm']
                    },
                    {
                        'priority': 'medium',
                        'impact_level': 'minor',  # Minor: small timing adjustment
                        'area': 'Transition',
                        'issue': 'Abrupt transition from backswing to downswing',
                        'suggestion': 'The transition should be smooth, not abrupt. Practice pausing briefly at the top of your backswing before starting the downswing.',
                        'drill': 'Swing to top, hold for 1 second, then smoothly start downswing. Repeat 20 times.',
                        'key_points': ['Pause at top', 'Smooth transition', 'No rushing']
                    }
                ],
                'optimization': [
                    {
                        'priority': 'low',
                        'impact_level': 'minor',
                        'area': 'Tempo Refinement',
                        'issue': 'Minor tempo inconsistencies',
                        'suggestion': 'Fine-tune your existing tempo. If your swing is generally good, focus on making your rhythm slightly more consistent without changing your natural pace.',
                        'drill': 'Practice maintaining your current tempo but with more consistency. Count in your head to maintain the same rhythm.',
                        'key_points': ['Maintain natural pace', 'Increase consistency', 'Small refinement']
                    }
                ]
            },
            'weight_shift': {
                'low': [
                    {
                        'priority': 'high',
                        'impact_level': 'major',  # Major: changes fundamental mechanics
                        'area': 'Weight Transfer',
                        'issue': 'Weight stuck on back foot',
                        'suggestion': 'Focus on shifting weight to your front foot during the downswing. The lower body should lead, with weight transferring 6 inches ahead of the upper body at impact. Practice feeling your weight move from back foot to front foot.',
                        'drill': 'Practice weight shift: Start with 60% weight on back foot, shift to 80% on front foot during downswing. Do this slowly, focusing on the feeling.',
                        'key_points': ['Weight shifts to front foot', 'Lower body leads', '6 inches ahead at impact']
                    },
                    {
                        'priority': 'high',
                        'impact_level': 'major',  # Major: changes sequence
                        'area': 'Lower Body Lead',
                        'issue': 'Upper body leading the downswing',
                        'suggestion': 'The downswing should start with the lower body (hips), not the upper body. Feel your hips rotate first, then torso, then arms. This creates proper sequence and power.',
                        'drill': 'Practice "bump and turn": Small hip bump toward target, then hip rotation. Upper body follows. Repeat 30 times slowly.',
                        'key_points': ['Hips start downswing', 'Lower body leads', 'Proper sequence']
                    }
                ],
                'optimization': [
                    {
                        'priority': 'low',
                        'impact_level': 'minor',
                        'area': 'Weight Shift Timing',
                        'issue': 'Minor weight shift timing issues',
                        'suggestion': 'Optimize the timing of your existing weight shift. If you already shift weight, focus on doing it slightly earlier or more smoothly without changing your overall mechanics.',
                        'drill': 'Practice your current swing but focus on starting the weight shift slightly earlier in the downswing.',
                        'key_points': ['Keep existing mechanics', 'Optimize timing', 'Small refinement']
                    }
                ]
            },
            'body_rotation': {
                'low': [
                    {
                        'priority': 'high',
                        'impact_level': 'major',  # Major: changes rotation mechanics
                        'area': 'Shoulder Turn',
                        'issue': 'Limited shoulder rotation',
                        'suggestion': 'Focus on making a full shoulder turn. For driver, aim for 90°+ shoulder turn. Keep your back to the target longer, allowing full rotation.',
                        'drill': 'Cross-armed drill: Fold arms across chest, practice full shoulder turn. Feel the stretch in your back. Repeat 20 times.',
                        'key_points': ['Full shoulder turn', '90°+ for driver', 'Back to target']
                    },
                    {
                        'priority': 'medium',
                        'impact_level': 'major',  # Major: changes sequence
                        'area': 'Kinematic Sequence',
                        'issue': 'Improper rotation sequence',
                        'suggestion': 'Proper sequence: Hips rotate first, then torso, then arms. Practice feeling this sequence - it should feel like a chain reaction, not all at once.',
                        'drill': 'Practice sequence slowly: Hips → Torso → Arms. Do this in slow motion 10 times, then gradually increase speed.',
                        'key_points': ['Hips first', 'Then torso', 'Then arms', 'Chain reaction']
                    }
                ],
                'optimization': [
                    {
                        'priority': 'low',
                        'impact_level': 'minor',
                        'area': 'Rotation Efficiency',
                        'issue': 'Minor rotation efficiency issues',
                        'suggestion': 'Optimize your existing rotation. If you already rotate, focus on making it slightly more efficient or consistent without changing your natural turn.',
                        'drill': 'Practice your current swing but focus on maintaining rotation through impact slightly longer.',
                        'key_points': ['Keep existing turn', 'Optimize efficiency', 'Small refinement']
                    }
                ]
            },
            'impact_quality': {
                'low': [
                    {
                        'priority': 'high',
                        'impact_level': 'major',  # Major: changes impact mechanics
                        'area': 'Impact Position',
                        'issue': 'Poor impact mechanics',
                        'suggestion': 'Impact is the moment of truth. Focus on proper sequence (weight forward, body rotated, hands ahead of clubhead). Practice impact position: weight on front foot, hands ahead, clubface square.',
                        'drill': 'Impact bag drill: Practice hitting an impact bag, focusing on proper impact position. Feel weight forward, hands ahead.',
                        'key_points': ['Weight forward', 'Hands ahead of clubhead', 'Square clubface']
                    },
                    {
                        'priority': 'high',
                        'impact_level': 'moderate',  # Moderate: affects speed but not mechanics
                        'area': 'Clubhead Speed',
                        'issue': 'Low clubhead speed at impact',
                        'suggestion': 'Clubhead speed comes from proper sequence and weight transfer. Focus on the fundamentals: full rotation, weight shift, and smooth tempo. Speed is a result, not a goal.',
                        'drill': 'Speed ladder: Start with 50% effort, focus on sequence. Gradually increase to 75%, then 90% while maintaining proper mechanics.',
                        'key_points': ['Proper sequence', 'Weight transfer', 'Smooth tempo', 'Speed is result']
                    }
                ],
                'optimization': [
                    {
                        'priority': 'low',
                        'impact_level': 'minor',
                        'area': 'Impact Consistency',
                        'issue': 'Minor impact inconsistencies',
                        'suggestion': 'Fine-tune your existing impact position. If your impact is generally good, focus on making it slightly more consistent without changing your mechanics.',
                        'drill': 'Practice your current swing but focus on maintaining the same impact position more consistently.',
                        'key_points': ['Keep existing mechanics', 'Increase consistency', 'Small refinement']
                    }
                ]
            },
            'followthrough': {
                'low': [
                    {
                        'priority': 'high',
                        'impact_level': 'moderate',  # Moderate: affects finish but not core mechanics
                        'area': 'Follow-Through',
                        'issue': 'Incomplete follow-through',
                        'suggestion': 'Complete your swing! Follow-through should be as long or longer than your backswing. Finish with weight fully on your front foot, belt buckle facing the target, and hold the finish position.',
                        'drill': 'Finish position drill: Swing and hold finish for 5 seconds. Check: Weight on front foot? Belt buckle facing target? Balanced?',
                        'key_points': ['Complete finish', 'Weight on front foot', 'Belt buckle to target', 'Hold position']
                    },
                    {
                        'priority': 'medium',
                        'impact_level': 'minor',  # Minor: small extension tweak
                        'area': 'Extension',
                        'issue': 'Lack of extension through impact',
                        'suggestion': 'Maintain extension through impact and into follow-through. Don\'t collapse or stop - let the club continue through the ball and finish high.',
                        'drill': 'Extension drill: Practice swinging through impact, focusing on full extension. Feel the club continue past impact.',
                        'key_points': ['Full extension', 'Continue through ball', 'Finish high']
                    }
                ],
                'optimization': [
                    {
                        'priority': 'low',
                        'impact_level': 'minor',
                        'area': 'Finish Position',
                        'issue': 'Minor finish position inconsistencies',
                        'suggestion': 'Optimize your existing finish. If you already finish well, focus on holding the finish position slightly longer or making it more consistent.',
                        'drill': 'Practice your current swing but hold the finish for 5 seconds consistently.',
                        'key_points': ['Keep existing finish', 'Increase consistency', 'Small refinement']
                    }
                ]
            },
            'balance': {
                'low': [
                    {
                        'priority': 'high',
                        'impact_level': 'moderate',  # Moderate: affects stability but not mechanics
                        'area': 'Balance',
                        'issue': 'Unstable throughout swing',
                        'suggestion': 'Balance is fundamental. Practice swinging while maintaining balance. Start with slow, controlled swings. Focus on staying centered and stable throughout.',
                        'drill': 'Balance drill: Swing slowly, focusing on staying balanced. If you lose balance, slow down. Gradually increase speed while maintaining balance.',
                        'key_points': ['Stay centered', 'Stable throughout', 'Start slow']
                    },
                    {
                        'priority': 'high',
                        'impact_level': 'minor',  # Minor: finish position tweak
                        'area': 'Finish Balance',
                        'issue': 'Cannot hold finish position',
                        'suggestion': 'A good swing ends in balance. Practice finishing your swing and holding the finish position for 5 seconds. If you can\'t hold it, your swing is too fast or off-balance.',
                        'drill': 'Finish hold drill: Every swing, hold finish for 5 seconds. If you fall, the swing was too aggressive. Adjust and repeat.',
                        'key_points': ['Hold finish', '5 seconds', 'Stable position']
                    }
                ],
                'optimization': [
                    {
                        'priority': 'low',
                        'impact_level': 'minor',
                        'area': 'Balance Refinement',
                        'issue': 'Minor balance inconsistencies',
                        'suggestion': 'Fine-tune your existing balance. If you\'re generally balanced, focus on maintaining balance slightly better through impact without changing your swing mechanics.',
                        'drill': 'Practice your current swing but focus on staying slightly more centered through impact.',
                        'key_points': ['Keep existing mechanics', 'Optimize balance', 'Small refinement']
                    }
                ]
            },
            'consistency': {
                'low': [
                    {
                        'priority': 'high',
                        'impact_level': 'moderate',  # Moderate: affects repeatability
                        'area': 'Consistency',
                        'issue': 'Inconsistent swing motion',
                        'suggestion': 'Consistency comes from repeatable fundamentals. Focus on the same setup, same tempo, same sequence every time. Practice the same swing, not different swings.',
                        'drill': 'Repetition drill: Hit 10 balls focusing on doing the EXACT same thing each time. Same setup, same tempo, same sequence.',
                        'key_points': ['Same setup', 'Same tempo', 'Same sequence', 'Repeatable']
                    },
                    {
                        'priority': 'medium',
                        'impact_level': 'minor',  # Minor: timing refinement
                        'area': 'Timing',
                        'issue': 'Inconsistent timing between phases',
                        'suggestion': 'Work on consistent timing. Use a metronome or count in your head to maintain the same rhythm for every swing.',
                        'drill': 'Metronome practice: Set metronome to 60 BPM. Backswing on beats 1-2, transition on 3, downswing on 4-5. Practice until consistent.',
                        'key_points': ['Consistent rhythm', 'Same timing', 'Metronome practice']
                    }
                ],
                'optimization': [
                    {
                        'priority': 'low',
                        'impact_level': 'minor',
                        'area': 'Consistency Refinement',
                        'issue': 'Minor consistency issues',
                        'suggestion': 'Fine-tune your existing swing consistency. If your swing is generally consistent, focus on making it slightly more repeatable without changing your mechanics.',
                        'drill': 'Practice your current swing but focus on doing the exact same thing 10 times in a row.',
                        'key_points': ['Keep existing mechanics', 'Increase repeatability', 'Small refinement']
                    }
                ]
            }
        }
    
    def analyze_swing(self, quality_scores: np.ndarray) -> Dict:
        """
        Analyze swing quality scores and identify areas for improvement.
        
        Args:
            quality_scores: 8-dimensional array [setup, tempo, weight_shift, rotation, 
                                                 impact, followthrough, balance, consistency]
        
        Returns:
            Analysis dictionary with weaknesses and strengths
        """
        score_names = [
            'setup_quality', 'tempo_rhythm', 'weight_shift', 'body_rotation',
            'impact_quality', 'followthrough', 'balance', 'consistency'
        ]
        
        scores_dict = dict(zip(score_names, quality_scores))
        
        # Identify weaknesses (below threshold)
        weaknesses = []
        strengths = []
        
        for name, score in scores_dict.items():
            threshold = self.thresholds[name]
            if score < threshold:
                weaknesses.append({
                    'area': name,
                    'score': float(score),
                    'threshold': threshold,
                    'gap': float(threshold - score)
                })
            else:
                strengths.append({
                    'area': name,
                    'score': float(score)
                })
        
        # Sort weaknesses by gap (biggest improvement opportunity first)
        weaknesses.sort(key=lambda x: x['gap'], reverse=True)
        
        return {
            'overall_score': float(np.mean(quality_scores)),
            'weaknesses': weaknesses,
            'strengths': strengths,
            'scores': scores_dict
        }
    
    def get_improvement_suggestions(
        self, 
        quality_scores: np.ndarray,
        max_suggestions: int = 5,
        prioritize: bool = True
    ) -> List[Dict]:
        """
        Get actionable improvement suggestions based on quality scores.
        
        Args:
            quality_scores: 8-dimensional quality array
            max_suggestions: Maximum number of suggestions to return
            prioritize: If True, prioritize by impact and feasibility
        
        Returns:
            List of improvement suggestions with priorities, drills, and key points
        """
        analysis = self.analyze_swing(quality_scores)
        
        suggestions = []
        score_names = [
            'setup_quality', 'tempo_rhythm', 'weight_shift', 'body_rotation',
            'impact_quality', 'followthrough', 'balance', 'consistency'
        ]
        
        # Get suggestions for each weakness
        for weakness in analysis['weaknesses']:
            area = weakness['area']
            score = weakness['score']
            
            # Determine severity
            threshold = self.thresholds[area]
            gap = threshold - score
            
            if gap > 0.3:
                severity = 'very_low'
            elif gap > 0.15:
                severity = 'low'
            else:
                severity = 'needs_improvement'
            
            # Check if this is a critical issue (only suggest major changes for critical)
            is_critical = score < self.critical_threshold or gap > self.major_change_gap_threshold
            
            # Get suggestions for this area
            if area in self.suggestions_db:
                # In conservative mode, prefer optimizations for non-critical issues
                if self.conservative_mode and not is_critical and 'optimization' in self.suggestions_db[area]:
                    # Add optimization suggestions first (minor tweaks)
                    opt_suggestions = self.suggestions_db[area]['optimization']
                    for suggestion in opt_suggestions:
                        enhanced_suggestion = suggestion.copy()
                        enhanced_suggestion['area'] = area
                        enhanced_suggestion['current_score'] = float(score)
                        enhanced_suggestion['target_score'] = float(threshold)
                        enhanced_suggestion['improvement_gap'] = float(gap)
                        enhanced_suggestion['is_optimization'] = True
                        suggestions.append(enhanced_suggestion)
                
                # Get regular suggestions
                area_suggestions = self.suggestions_db[area].get(severity, 
                                                                self.suggestions_db[area].get('low', []))
                
                for suggestion in area_suggestions:
                    # In conservative mode, filter out major changes for non-critical issues
                    impact_level = suggestion.get('impact_level', 'moderate')
                    if self.conservative_mode and not is_critical and impact_level == 'major':
                        continue  # Skip major changes for non-critical issues
                    
                    # Add context
                    enhanced_suggestion = suggestion.copy()
                    enhanced_suggestion['area'] = area
                    enhanced_suggestion['current_score'] = float(score)
                    enhanced_suggestion['target_score'] = float(threshold)
                    enhanced_suggestion['improvement_gap'] = float(gap)
                    enhanced_suggestion['is_critical'] = is_critical
                    suggestions.append(enhanced_suggestion)
        
        # Prioritize suggestions
        if prioritize:
            # In conservative mode, prioritize: minor > moderate > major (for non-critical)
            # For critical issues, prioritize by gap and priority
            def get_sort_key(x):
                if self.conservative_mode and not x.get('is_critical', False):
                    # Prefer minor optimizations
                    impact_priority = {'minor': 0, 'moderate': 1, 'major': 2}.get(
                        x.get('impact_level', 'moderate'), 1
                    )
                    is_opt = x.get('is_optimization', False)
                    opt_boost = -0.5 if is_opt else 0  # Boost optimizations
                    priority_val = {'high': 0, 'medium': 1, 'low': 2}.get(x.get('priority', 'medium'), 1)
                    return (impact_priority + opt_boost, priority_val, -x.get('improvement_gap', 0))
                else:
                    # Critical issues: prioritize by gap and priority
                    priority_val = {'high': 0, 'medium': 1, 'low': 2}.get(x.get('priority', 'medium'), 1)
                    return (priority_val, -x.get('improvement_gap', 0))
            
            suggestions.sort(key=get_sort_key)
        
        # Limit number of suggestions
        suggestions = suggestions[:max_suggestions]
        
        return suggestions
    
    def generate_improvement_report(
        self,
        quality_scores: np.ndarray,
        video_id: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive improvement report.
        
        Args:
            quality_scores: 8-dimensional quality array
            video_id: Optional video identifier
        
        Returns:
            Complete improvement report
        """
        analysis = self.analyze_swing(quality_scores)
        suggestions = self.get_improvement_suggestions(quality_scores)
        
        report = {
            'video_id': video_id,
            'overall_score': analysis['overall_score'],
            'grade': self._get_grade(analysis['overall_score']),
            'summary': self._generate_summary(analysis),
            'strengths': analysis['strengths'],
            'weaknesses': analysis['weaknesses'],
            'suggestions': suggestions,
            'action_plan': self._create_action_plan(suggestions)
        }
        
        return report
    
    def _get_grade(self, overall_score: float) -> str:
        """Convert overall score to letter grade"""
        if overall_score >= 0.90:
            return 'A+'
        elif overall_score >= 0.85:
            return 'A'
        elif overall_score >= 0.80:
            return 'A-'
        elif overall_score >= 0.75:
            return 'B+'
        elif overall_score >= 0.70:
            return 'B'
        elif overall_score >= 0.65:
            return 'B-'
        elif overall_score >= 0.60:
            return 'C+'
        elif overall_score >= 0.55:
            return 'C'
        elif overall_score >= 0.50:
            return 'C-'
        else:
            return 'D'
    
    def _generate_summary(self, analysis: Dict) -> str:
        """Generate human-readable summary"""
        overall = analysis['overall_score']
        num_weaknesses = len(analysis['weaknesses'])
        num_strengths = len(analysis['strengths'])
        
        if overall >= 0.80:
            summary = f"Excellent swing! Overall score: {overall:.1%}. "
        elif overall >= 0.70:
            summary = f"Good swing with room for improvement. Overall score: {overall:.1%}. "
        elif overall >= 0.60:
            summary = f"Fair swing with several areas to work on. Overall score: {overall:.1%}. "
        else:
            summary = f"Swing needs significant improvement. Overall score: {overall:.1%}. "
        
        if num_weaknesses > 0:
            top_weakness = analysis['weaknesses'][0]
            summary += f"Main area to focus on: {top_weakness['area'].replace('_', ' ').title()} (score: {top_weakness['score']:.1%}). "
        
        if num_strengths > 0:
            summary += f"You have {num_strengths} strong area(s) to build upon."
        
        return summary
    
    def _create_action_plan(self, suggestions: List[Dict]) -> Dict:
        """Create prioritized action plan from suggestions"""
        if not suggestions:
            return {
                'immediate_focus': None,
                'this_week': [],
                'this_month': []
            }
        
        # Immediate focus: highest priority suggestion
        immediate = suggestions[0] if suggestions else None
        
        # This week: high priority suggestions
        this_week = [s for s in suggestions[:3] if s.get('priority') == 'high']
        
        # This month: all suggestions
        this_month = suggestions[:5]
        
        return {
            'immediate_focus': immediate,
            'this_week': this_week,
            'this_month': this_month
        }
    
    def record_feedback(
        self,
        suggestion: Dict,
        score_before: float,
        score_after: float,
        improvement: float
    ):
        """
        Record feedback on suggestion effectiveness (for learning).
        
        Args:
            suggestion: The suggestion that was followed
            score_before: Quality score before implementing suggestion
            score_after: Quality score after implementing suggestion
            improvement: Improvement amount (score_after - score_before)
        """
        if self.learn_from_feedback:
            self.suggestion_history.append({
                'suggestion': suggestion,
                'score_before': score_before,
                'score_after': score_after,
                'improvement': improvement
            })
    
    def get_effective_suggestions(self, min_improvement: float = 0.05) -> List[Dict]:
        """
        Get suggestions that have been most effective (if learning enabled).
        
        Args:
            min_improvement: Minimum improvement to consider effective
        
        Returns:
            List of effective suggestions sorted by average improvement
        """
        if not self.learn_from_feedback or not self.suggestion_history:
            return []
        
        # Group by suggestion type
        suggestion_effectiveness = {}
        
        for record in self.suggestion_history:
            suggestion_key = record['suggestion'].get('area', 'unknown')
            if suggestion_key not in suggestion_effectiveness:
                suggestion_effectiveness[suggestion_key] = {
                    'improvements': [],
                    'count': 0
                }
            
            if record['improvement'] >= min_improvement:
                suggestion_effectiveness[suggestion_key]['improvements'].append(record['improvement'])
                suggestion_effectiveness[suggestion_key]['count'] += 1
        
        # Calculate average improvements
        effective = []
        for area, data in suggestion_effectiveness.items():
            if data['improvements']:
                avg_improvement = np.mean(data['improvements'])
                effective.append({
                    'area': area,
                    'average_improvement': avg_improvement,
                    'success_count': data['count'],
                    'success_rate': data['count'] / len(self.suggestion_history)
                })
        
        # Sort by average improvement
        effective.sort(key=lambda x: x['average_improvement'], reverse=True)
        
        return effective


def generate_improvement_report_from_video(
    video_path: str,
    grader_model_path: str = 'models/golf_swing_grader_final.pth.tar',
    output_path: Optional[str] = None
):
    """
    Complete pipeline: Grade swing → Generate improvement report.
    
    Args:
        video_path: Path to golf swing video
        grader_model_path: Path to trained grader model
        output_path: Optional path to save report JSON
    
    Returns:
        Improvement report dictionary
    """
    # Load grader model and predict scores
    from golf_swing_grader import GolfSwingGrader, EventDetector
    import torch
    from test_video import SampleVideo
    from torchvision import transforms
    
    # Load model
    event_detector = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False
    )
    
    model = GolfSwingGrader(event_detector)
    checkpoint = torch.load(grader_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.cuda()
    
    # Load and process video
    dataset = SampleVideo(
        video_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    sample = dataset[0]
    frames = sample['images'].unsqueeze(0).cuda()
    
    # Predict scores
    with torch.no_grad():
        scores, attention = model(frames)
        scores = scores[0].cpu().numpy()
    
    # Generate improvement report
    advisor = SwingImprovementAdvisor()
    report = advisor.generate_improvement_report(scores, video_id=video_path)
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Improvement report saved to: {output_path}")
    
    return report


if __name__ == '__main__':
    # Example usage
    advisor = SwingImprovementAdvisor()
    
    # Example scores (poor swing)
    example_scores = np.array([
        0.65,  # Setup: Fair
        0.55,  # Tempo: Poor
        0.50,  # Weight Shift: Poor
        0.60,  # Rotation: Fair
        0.55,  # Impact: Poor
        0.50,  # Follow-through: Poor
        0.70,  # Balance: Fair
        0.45   # Consistency: Poor
    ])
    
    # Generate report
    report = advisor.generate_improvement_report(example_scores, video_id="example_swing")
    
    # Print report
    print("=" * 60)
    print("GOLF SWING IMPROVEMENT REPORT")
    print("=" * 60)
    print(f"\nOverall Score: {report['overall_score']:.1%} ({report['grade']})")
    print(f"\nSummary: {report['summary']}")
    print(f"\nTop 3 Improvement Suggestions:")
    for i, suggestion in enumerate(report['suggestions'][:3], 1):
        print(f"\n{i}. {suggestion['area'].replace('_', ' ').title()}")
        print(f"   Issue: {suggestion['issue']}")
        print(f"   Suggestion: {suggestion['suggestion']}")
        print(f"   Drill: {suggestion['drill']}")
        print(f"   Key Points: {', '.join(suggestion['key_points'])}")

