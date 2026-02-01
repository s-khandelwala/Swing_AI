"""
Adaptive Golf Swing Improvement System

Automatically learns which suggestions work best WITHOUT manual feedback.
Uses automatic feedback from re-graded swings to adapt and personalize.

Key Innovation: Automatic feedback loop
- Golfer gets suggestion
- Golfer practices and records new swing
- System automatically re-grades new swing
- System measures improvement
- System learns: "This suggestion worked for this type of swing"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import pickle
from collections import defaultdict
from swing_improvement_system import SwingImprovementAdvisor


class AdaptiveSwingAdvisor(SwingImprovementAdvisor):
    """
    Adaptive improvement advisor that learns automatically from swing improvements.
    
    No manual feedback needed - automatically tracks:
    - Which suggestions correlate with improvements
    - Which suggestions work for which swing types
    - Personalizes recommendations based on learned patterns
    """
    
    def __init__(
        self,
        learn_from_improvements=True,
        model_path='models/adaptive_advisor.pkl',
        conservative_mode=True,
        user_id: Optional[str] = None,
        user_weight: float = 5.0
    ):
        """
        Args:
            learn_from_improvements: Automatically learn from swing improvements
            model_path: Path to save/load learned patterns
            conservative_mode: If True, prioritizes minor optimizations over major changes
            user_id: Optional user identifier for person-specific learning
            user_weight: Weight for user-specific patterns vs global (default: 5.0 = 5x more important)
        """
        super().__init__(learn_from_feedback=False, conservative_mode=conservative_mode)  # We'll handle learning differently
        self.learn_from_improvements = learn_from_improvements
        self.model_path = model_path
        self.user_id = user_id
        self.user_weight = user_weight  # How much more to weight user-specific patterns
        
        # Learned patterns structure: {user_id: {area: pattern_data}}
        # Special key 'global' for global patterns (all users)
        # Each user_id has their own patterns
        self.effectiveness_patterns = {
            'global': defaultdict(lambda: {
                'total_uses': 0,
                'successful_improvements': 0,
                'avg_improvement': 0.0,
                'swing_type_patterns': defaultdict(lambda: {'count': 0, 'improvements': []})
            })
        }
        
        # Load existing patterns if available
        if learn_from_improvements:
            self._load_patterns()
    
    def _load_patterns(self):
        """Load learned effectiveness patterns (dictionary with user_id as keys)"""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                # Load dictionary structure: {user_id: patterns}
                loaded_patterns = data.get('patterns', {})
                
                # Ensure 'global' key exists
                if 'global' not in loaded_patterns:
                    loaded_patterns['global'] = defaultdict(lambda: {
                        'total_uses': 0,
                        'successful_improvements': 0,
                        'avg_improvement': 0.0,
                        'swing_type_patterns': defaultdict(lambda: {'count': 0, 'improvements': []})
                    })
                
                self.effectiveness_patterns = loaded_patterns
        except FileNotFoundError:
            # Start fresh with global patterns
            self.effectiveness_patterns = {
                'global': defaultdict(lambda: {
                    'total_uses': 0,
                    'successful_improvements': 0,
                    'avg_improvement': 0.0,
                    'swing_type_patterns': defaultdict(lambda: {'count': 0, 'improvements': []})
                })
            }
    
    def _save_patterns(self):
        """Save learned effectiveness patterns (dictionary with user_id as keys)"""
        import os
        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else '.', exist_ok=True)
        
        # Convert defaultdicts to regular dicts for serialization
        serializable_patterns = {}
        for user_id, user_patterns in self.effectiveness_patterns.items():
            serializable_patterns[user_id] = dict(user_patterns)
        
        data = {'patterns': serializable_patterns}
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _classify_swing_type(self, quality_scores: np.ndarray) -> str:
        """
        Classify swing type based on score patterns.
        This helps personalize suggestions.
        
        The system identifies common weakness patterns:
        - power_issues: weight_shift + body_rotation weak
        - rhythm_issues: tempo_rhythm + consistency weak
        - stability_issues: balance + followthrough weak
        - fundamentals_issues: setup_quality weak (primary)
        - mixed_issues: Multiple different weaknesses (default)
        
        Returns: Swing type identifier
        """
        score_names = [
            'setup_quality', 'tempo_rhythm', 'weight_shift', 'body_rotation',
            'impact_quality', 'followthrough', 'balance', 'consistency'
        ]
        
        # Identify weaknesses (scores below threshold)
        weaknesses = []
        for i, (name, score) in enumerate(zip(score_names, quality_scores)):
            threshold = self.thresholds.get(name, 0.70)
            if score < threshold:
                weaknesses.append(score_names.index(name))
        
        # Convert to names for easier pattern matching
        weakness_names = [score_names[i] for i in weaknesses]
        
        # Pattern 1: Power Issues
        # Weak weight shift + weak rotation = power generation problems
        if 'weight_shift' in weakness_names and 'body_rotation' in weakness_names:
            return 'power_issues'
        
        # Pattern 2: Rhythm Issues
        # Weak tempo + weak consistency = timing/rhythm problems
        if 'tempo_rhythm' in weakness_names and 'consistency' in weakness_names:
            return 'rhythm_issues'
        
        # Pattern 3: Stability Issues
        # Weak balance + weak followthrough = stability problems
        if 'balance' in weakness_names and 'followthrough' in weakness_names:
            return 'stability_issues'
        
        # Pattern 4: Fundamentals Issues
        # Weak setup = fundamental problems (affects everything)
        if 'setup_quality' in weakness_names:
            return 'fundamentals_issues'
        
        # Pattern 5: Mixed Issues (default)
        # Multiple different weaknesses, no clear pattern
        return 'mixed_issues'
    
    def get_adaptive_suggestions(
        self,
        quality_scores: np.ndarray,
        max_suggestions: int = 5,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get suggestions personalized based on learned effectiveness.
        
        Automatically adapts based on:
        - Which suggestions have worked for similar swings
        - Swing type classification
        - Historical improvement patterns
        - User-specific patterns (weighted much more heavily)
        - Global patterns (from all users)
        
        Args:
            quality_scores: 8-dimensional quality array
            max_suggestions: Maximum number of suggestions
            user_id: Optional user identifier (uses self.user_id if not provided)
        """
        # Use provided user_id or instance user_id
        current_user_id = user_id if user_id is not None else self.user_id
        
        # Get base suggestions (rule-based)
        base_suggestions = self.get_improvement_suggestions(quality_scores, max_suggestions * 2)
        
        # Classify swing type
        swing_type = self._classify_swing_type(quality_scores)
        
        # Enhance suggestions with learned effectiveness (hybrid: user + global)
        enhanced_suggestions = []
        for suggestion in base_suggestions:
            enhanced = suggestion.copy()
            area = suggestion['area']
            
            # Get user-specific patterns (if available)
            user_pattern = None
            if current_user_id and current_user_id in self.effectiveness_patterns:
                user_patterns = self.effectiveness_patterns[current_user_id]
                if area in user_patterns:
                    user_pattern = user_patterns[area]
            
            # Get global patterns (always available)
            global_pattern = None
            if 'global' in self.effectiveness_patterns:
                global_patterns = self.effectiveness_patterns['global']
                if area in global_patterns:
                    global_pattern = global_patterns[area]
            
            # Combine user-specific (weighted heavily) + global patterns
            if user_pattern or global_pattern:
                # Calculate weighted effectiveness
                user_success_rate = None
                user_avg_improvement = None
                user_type_success_rate = None
                user_type_avg_improvement = None
                
                if user_pattern and user_pattern['total_uses'] > 0:
                    user_success_rate = user_pattern['successful_improvements'] / user_pattern['total_uses']
                    user_avg_improvement = user_pattern['avg_improvement']
                    
                    if swing_type in user_pattern['swing_type_patterns']:
                        type_pattern = user_pattern['swing_type_patterns'][swing_type]
                        if type_pattern['count'] > 0:
                            type_improvements = type_pattern['improvements']
                            user_type_success_rate = len([i for i in type_improvements if i > 0.05]) / len(type_improvements)
                            user_type_avg_improvement = np.mean(type_improvements) if type_improvements else 0
                
                global_success_rate = None
                global_avg_improvement = None
                global_type_success_rate = None
                global_type_avg_improvement = None
                
                if global_pattern and global_pattern['total_uses'] > 0:
                    global_success_rate = global_pattern['successful_improvements'] / global_pattern['total_uses']
                    global_avg_improvement = global_pattern['avg_improvement']
                    
                    if swing_type in global_pattern['swing_type_patterns']:
                        type_pattern = global_pattern['swing_type_patterns'][swing_type]
                        if type_pattern['count'] > 0:
                            type_improvements = type_pattern['improvements']
                            global_type_success_rate = len([i for i in type_improvements if i > 0.05]) / len(type_improvements)
                            global_type_avg_improvement = np.mean(type_improvements) if type_improvements else 0
                
                # Weighted combination: user patterns weighted much more heavily
                if user_success_rate is not None and global_success_rate is not None:
                    # Weighted average: user_weight * user + 1 * global
                    total_weight = self.user_weight + 1.0
                    enhanced['learned_success_rate'] = (
                        (self.user_weight * user_success_rate + 1.0 * global_success_rate) / total_weight
                    )
                    enhanced['learned_avg_improvement'] = (
                        (self.user_weight * user_avg_improvement + 1.0 * global_avg_improvement) / total_weight
                    )
                elif user_success_rate is not None:
                    # Only user data available
                    enhanced['learned_success_rate'] = user_success_rate
                    enhanced['learned_avg_improvement'] = user_avg_improvement
                elif global_success_rate is not None:
                    # Only global data available
                    enhanced['learned_success_rate'] = global_success_rate
                    enhanced['learned_avg_improvement'] = global_avg_improvement
                else:
                    enhanced['learned_success_rate'] = None
                    enhanced['learned_avg_improvement'] = None
                
                # Swing type specific (weighted)
                if user_type_success_rate is not None and global_type_success_rate is not None:
                    total_weight = self.user_weight + 1.0
                    enhanced['swing_type_success_rate'] = (
                        (self.user_weight * user_type_success_rate + 1.0 * global_type_success_rate) / total_weight
                    )
                    enhanced['swing_type_avg_improvement'] = (
                        (self.user_weight * user_type_avg_improvement + 1.0 * global_type_avg_improvement) / total_weight
                    )
                elif user_type_success_rate is not None:
                    enhanced['swing_type_success_rate'] = user_type_success_rate
                    enhanced['swing_type_avg_improvement'] = user_type_avg_improvement
                elif global_type_success_rate is not None:
                    enhanced['swing_type_success_rate'] = global_type_success_rate
                    enhanced['swing_type_avg_improvement'] = global_type_avg_improvement
                else:
                    enhanced['swing_type_success_rate'] = None
                    enhanced['swing_type_avg_improvement'] = None
                
                # Store source info for debugging
                enhanced['learning_source'] = 'user+global' if (user_pattern and global_pattern) else ('user' if user_pattern else 'global')
            else:
                # No learning yet - use base priority
                enhanced['learned_success_rate'] = None
                enhanced['learned_avg_improvement'] = None
                enhanced['swing_type_success_rate'] = None
                enhanced['swing_type_avg_improvement'] = None
                enhanced['learning_source'] = 'none'
            
            enhanced_suggestions.append(enhanced)
        
        # Re-prioritize based on learned effectiveness
        enhanced_suggestions.sort(key=lambda x: self._get_adaptive_priority(x), reverse=True)
        
        return enhanced_suggestions[:max_suggestions]
    
    def _get_adaptive_priority(self, suggestion: Dict) -> float:
        """
        Calculate adaptive priority score.
        Combines base priority with learned effectiveness.
        Prefers minor optimizations over major changes (unless critical).
        """
        base_priority = {'high': 3, 'medium': 2, 'low': 1}.get(suggestion.get('priority', 'medium'), 2)
        
        # Prefer minor optimizations (less disruptive)
        impact_level = suggestion.get('impact_level', 'moderate')
        impact_boost = {'minor': 1.5, 'moderate': 0.5, 'major': -0.5}.get(impact_level, 0)
        
        # Boost optimizations (fine-tuning existing swing)
        if suggestion.get('is_optimization', False):
            impact_boost += 1.0
        
        # Only apply impact boost for non-critical issues
        if not suggestion.get('is_critical', False):
            base_priority += impact_boost
        
        # Boost if we've learned this works well
        learned_boost = 0
        if suggestion.get('learned_success_rate') is not None:
            # High success rate = boost priority
            learned_boost = suggestion['learned_success_rate'] * 2
        
        # Boost if it works well for this swing type
        type_boost = 0
        if suggestion.get('swing_type_success_rate') is not None:
            type_boost = suggestion['swing_type_success_rate'] * 1.5
        
        # Penalize if we've learned it doesn't work
        penalty = 0
        if suggestion.get('learned_success_rate') is not None and suggestion['learned_success_rate'] < 0.3:
            penalty = -1  # Reduce priority if low success rate
        
        return base_priority + learned_boost + type_boost + penalty
    
    def record_automatic_feedback(
        self,
        initial_scores: np.ndarray,
        improved_scores: np.ndarray,
        suggestions_given: List[Dict],
        swing_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Automatically record feedback from swing improvement.
        
        No manual input needed - system automatically:
        1. Measures improvement in each area
        2. Correlates improvements with suggestions given
        3. Learns which suggestions work
        4. Updates BOTH user-specific and global patterns
        
        Args:
            initial_scores: Quality scores before suggestions
            improved_scores: Quality scores after practicing
            suggestions_given: List of suggestions that were provided
            swing_id: Optional identifier
            user_id: Optional user identifier (uses self.user_id if not provided)
        """
        if not self.learn_from_improvements:
            return
        
        # Use provided user_id or instance user_id
        current_user_id = user_id if user_id is not None else self.user_id
        
        # Calculate improvements
        improvements = improved_scores - initial_scores
        
        # Classify swing type
        swing_type = self._classify_swing_type(initial_scores)
        
        # Update effectiveness patterns for each suggestion
        score_names = [
            'setup_quality', 'tempo_rhythm', 'weight_shift', 'body_rotation',
            'impact_quality', 'followthrough', 'balance', 'consistency'
        ]
        
        for suggestion in suggestions_given:
            area = suggestion.get('area')
            if area not in score_names:
                continue
            
            area_idx = score_names.index(area)
            area_improvement = improvements[area_idx]
            
            # Update GLOBAL patterns (all users)
            if 'global' not in self.effectiveness_patterns:
                self.effectiveness_patterns['global'] = defaultdict(lambda: {
                    'total_uses': 0,
                    'successful_improvements': 0,
                    'avg_improvement': 0.0,
                    'swing_type_patterns': defaultdict(lambda: {'count': 0, 'improvements': []})
                })
            
            global_pattern = self.effectiveness_patterns['global'][area]
            global_pattern['total_uses'] += 1
            if area_improvement > 0.05:
                global_pattern['successful_improvements'] += 1
            current_avg = global_pattern['avg_improvement']
            total = global_pattern['total_uses']
            global_pattern['avg_improvement'] = (current_avg * (total - 1) + area_improvement) / total
            
            # Update global swing type pattern
            global_type_pattern = global_pattern['swing_type_patterns'][swing_type]
            global_type_pattern['count'] += 1
            global_type_pattern['improvements'].append(area_improvement)
            if len(global_type_pattern['improvements']) > 100:
                global_type_pattern['improvements'] = global_type_pattern['improvements'][-100:]
            
            # Update USER-SPECIFIC patterns (if user_id provided)
            if current_user_id:
                if current_user_id not in self.effectiveness_patterns:
                    self.effectiveness_patterns[current_user_id] = defaultdict(lambda: {
                        'total_uses': 0,
                        'successful_improvements': 0,
                        'avg_improvement': 0.0,
                        'swing_type_patterns': defaultdict(lambda: {'count': 0, 'improvements': []})
                    })
                
                user_pattern = self.effectiveness_patterns[current_user_id][area]
                user_pattern['total_uses'] += 1
                if area_improvement > 0.05:
                    user_pattern['successful_improvements'] += 1
                current_avg = user_pattern['avg_improvement']
                total = user_pattern['total_uses']
                user_pattern['avg_improvement'] = (current_avg * (total - 1) + area_improvement) / total
                
                # Update user swing type pattern
                user_type_pattern = user_pattern['swing_type_patterns'][swing_type]
                user_type_pattern['count'] += 1
                user_type_pattern['improvements'].append(area_improvement)
                if len(user_type_pattern['improvements']) > 100:
                    user_type_pattern['improvements'] = user_type_pattern['improvements'][-100:]
                
                # Track impact level effectiveness for user
                impact_level = suggestion.get('impact_level', 'moderate')
                if 'impact_level_patterns' not in user_pattern:
                    user_pattern['impact_level_patterns'] = defaultdict(lambda: {'count': 0, 'improvements': []})
                user_impact_pattern = user_pattern['impact_level_patterns'][impact_level]
                user_impact_pattern['count'] += 1
                user_impact_pattern['improvements'].append(area_improvement)
                if len(user_impact_pattern['improvements']) > 50:
                    user_impact_pattern['improvements'] = user_impact_pattern['improvements'][-50:]
            
            # Track impact level effectiveness for global
            impact_level = suggestion.get('impact_level', 'moderate')
            if 'impact_level_patterns' not in global_pattern:
                global_pattern['impact_level_patterns'] = defaultdict(lambda: {'count': 0, 'improvements': []})
            global_impact_pattern = global_pattern['impact_level_patterns'][impact_level]
            global_impact_pattern['count'] += 1
            global_impact_pattern['improvements'].append(area_improvement)
            if len(global_impact_pattern['improvements']) > 50:
                global_impact_pattern['improvements'] = global_impact_pattern['improvements'][-50:]
        
        # Save learned patterns
        self._save_patterns()
    
    def get_improvement_report_adaptive(
        self,
        quality_scores: np.ndarray,
        video_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict:
        """
        Generate improvement report with adaptive suggestions.
        
        Args:
            quality_scores: 8-dimensional quality array
            video_id: Optional video identifier
            user_id: Optional user identifier (uses self.user_id if not provided)
        """
        analysis = self.analyze_swing(quality_scores)
        current_user_id = user_id if user_id is not None else self.user_id
        suggestions = self.get_adaptive_suggestions(quality_scores, user_id=current_user_id)
        swing_type = self._classify_swing_type(quality_scores)
        
        report = {
            'video_id': video_id,
            'overall_score': analysis['overall_score'],
            'grade': self._get_grade(analysis['overall_score']),
            'swing_type': swing_type,
            'swing_type_explanation': self._get_swing_type_explanation(swing_type),
            'summary': self._generate_summary(analysis),
            'strengths': analysis['strengths'],
            'weaknesses': analysis['weaknesses'],
            'suggestions': suggestions,
            'action_plan': self._create_action_plan(suggestions),
            'learning_status': self._get_learning_status(user_id=current_user_id)
        }
        
        return report
    
    def _get_swing_type_explanation(self, swing_type: str) -> str:
        """Get human-readable explanation of swing type"""
        explanations = {
            'power_issues': 'Your swing shows power generation issues. Focus on weight shift and body rotation to generate more power.',
            'rhythm_issues': 'Your swing shows timing and rhythm issues. Focus on tempo and consistency to improve repeatability.',
            'stability_issues': 'Your swing shows stability issues. Focus on balance and follow-through to create a more stable swing.',
            'fundamentals_issues': 'Your swing shows fundamental setup issues. Focus on improving your stance, posture, and alignment first.',
            'mixed_issues': 'Your swing shows multiple different areas for improvement. Focus on the highest priority suggestions.'
        }
        return explanations.get(swing_type, 'Multiple areas for improvement.')
    
    def _get_learning_status(self, user_id: Optional[str] = None) -> Dict:
        """Get status of learned patterns"""
        current_user_id = user_id if user_id is not None else self.user_id
        
        # Get patterns to analyze
        patterns_to_analyze = {}
        if current_user_id and current_user_id in self.effectiveness_patterns:
            patterns_to_analyze.update(self.effectiveness_patterns[current_user_id])
        if 'global' in self.effectiveness_patterns:
            patterns_to_analyze.update(self.effectiveness_patterns['global'])
        
        total_areas = len(patterns_to_analyze)
        areas_with_data = sum(1 for p in patterns_to_analyze.values() if isinstance(p, dict) and p.get('total_uses', 0) > 0)
        
        total_instances = sum(
            p.get('total_uses', 0) for p in patterns_to_analyze.values()
            if isinstance(p, dict)
        )
        
        return {
            'user_id': current_user_id,
            'total_learning_instances': total_instances,
            'areas_with_learning': areas_with_data,
            'most_effective_areas': self._get_most_effective_areas(user_id=current_user_id),
            'has_user_specific_data': current_user_id is not None and current_user_id in self.effectiveness_patterns,
            'has_global_data': 'global' in self.effectiveness_patterns
        }
    
    def _get_most_effective_areas(self, top_n: int = 3, user_id: Optional[str] = None) -> List[Dict]:
        """Get areas where suggestions have been most effective (hybrid: user + global)"""
        current_user_id = user_id if user_id is not None else self.user_id
        
        effectiveness = []
        
        # Get all unique areas from both user and global
        all_areas = set()
        if current_user_id and current_user_id in self.effectiveness_patterns:
            all_areas.update(self.effectiveness_patterns[current_user_id].keys())
        if 'global' in self.effectiveness_patterns:
            all_areas.update(self.effectiveness_patterns['global'].keys())
        
        for area in all_areas:
            # Get user pattern
            user_pattern = None
            if current_user_id and current_user_id in self.effectiveness_patterns:
                user_patterns = self.effectiveness_patterns[current_user_id]
                if area in user_patterns:
                    user_pattern = user_patterns[area]
            
            # Get global pattern
            global_pattern = None
            if 'global' in self.effectiveness_patterns:
                global_patterns = self.effectiveness_patterns['global']
                if area in global_patterns:
                    global_pattern = global_patterns[area]
            
            # Calculate weighted effectiveness
            user_success_rate = None
            user_uses = 0
            if user_pattern and user_pattern.get('total_uses', 0) > 5:
                user_uses = user_pattern['total_uses']
                user_success_rate = user_pattern['successful_improvements'] / user_uses
            
            global_success_rate = None
            global_uses = 0
            if global_pattern and global_pattern.get('total_uses', 0) > 5:
                global_uses = global_pattern['total_uses']
                global_success_rate = global_pattern['successful_improvements'] / global_uses
            
            # Weighted combination
            if user_success_rate is not None and global_success_rate is not None:
                total_weight = self.user_weight + 1.0
                combined_success_rate = (
                    (self.user_weight * user_success_rate + 1.0 * global_success_rate) / total_weight
                )
                total_uses = user_uses + global_uses
            elif user_success_rate is not None:
                combined_success_rate = user_success_rate
                total_uses = user_uses
            elif global_success_rate is not None:
                combined_success_rate = global_success_rate
                total_uses = global_uses
            else:
                continue
            
            effectiveness.append({
                'area': area,
                'success_rate': combined_success_rate,
                'total_uses': total_uses,
                'user_uses': user_uses,
                'global_uses': global_uses,
                'source': 'user+global' if (user_pattern and global_pattern) else ('user' if user_pattern else 'global')
            })
        
        effectiveness.sort(key=lambda x: x['success_rate'], reverse=True)
        return effectiveness[:top_n]


def automatic_improvement_tracking(
    initial_video_path: str,
    improved_video_path: str,
    grader_model_path: str,
    advisor: AdaptiveSwingAdvisor,
    suggestions_given: List[Dict]
):
    """
    Complete automatic feedback loop.
    
    1. Grade initial swing
    2. Get suggestions
    3. (Golfer practices)
    4. Grade improved swing
    5. Automatically learn which suggestions worked
    
    No manual feedback needed!
    """
    from golf_swing_grader import GolfSwingGrader, EventDetector
    import torch
    from test_video import SampleVideo
    from torchvision import transforms
    
    # Load grader model
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
    
    def grade_video(video_path):
        dataset = SampleVideo(
            video_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
        sample = dataset[0]
        frames = sample['images'].unsqueeze(0).cuda()
        
        with torch.no_grad():
            scores, _ = model(frames)
            return scores[0].cpu().numpy()
    
    # Grade both swings
    initial_scores = grade_video(initial_video_path)
    improved_scores = grade_video(improved_video_path)
    
    # Automatically record feedback
    advisor.record_automatic_feedback(
        initial_scores=initial_scores,
        improved_scores=improved_scores,
        suggestions_given=suggestions_given,
        swing_id=initial_video_path
    )
    
    # Calculate and return improvements
    improvements = improved_scores - initial_scores
    
    return {
        'initial_scores': initial_scores.tolist(),
        'improved_scores': improved_scores.tolist(),
        'improvements': improvements.tolist(),
        'overall_improvement': float(np.mean(improvements)),
        'swing_type': advisor._classify_swing_type(initial_scores)
    }


if __name__ == '__main__':
    # Example: Hybrid learning (user-specific + global)
    user_id = "golfer_123"
    advisor = AdaptiveSwingAdvisor(
        learn_from_improvements=True,
        user_id=user_id,
        user_weight=5.0  # User patterns weighted 5x more than global
    )
    
    # Initial swing
    initial_scores = np.array([0.65, 0.55, 0.50, 0.60, 0.55, 0.50, 0.70, 0.45])
    
    # Get suggestions (uses global patterns if available, otherwise rule-based)
    suggestions = advisor.get_adaptive_suggestions(initial_scores, user_id=user_id)
    print("Initial Suggestions (no learning yet):")
    for s in suggestions[:3]:
        print(f"  - {s['area']}: {s['suggestion'][:50]}...")
        print(f"    Learning source: {s.get('learning_source', 'none')}")
    
    # Classify swing type
    swing_type = advisor._classify_swing_type(initial_scores)
    print(f"\nSwing Type: {swing_type}")
    print(f"Explanation: {advisor._get_swing_type_explanation(swing_type)}")
    
    # Simulate: Golfer practices, records new swing
    # (In real use, this would be automatic)
    improved_scores = np.array([0.70, 0.60, 0.65, 0.65, 0.60, 0.55, 0.75, 0.50])
    
    # Automatically record feedback (updates BOTH user-specific and global patterns)
    advisor.record_automatic_feedback(
        initial_scores=initial_scores,
        improved_scores=improved_scores,
        suggestions_given=suggestions[:3],  # Top 3 suggestions
        user_id=user_id
    )
    
    print("\nAfter learning from improvement:")
    new_suggestions = advisor.get_adaptive_suggestions(initial_scores, user_id=user_id)
    for s in new_suggestions[:3]:
        success_rate = s.get('learned_success_rate')
        source = s.get('learning_source', 'none')
        if success_rate is not None:
            print(f"  - {s['area']}: Success rate: {success_rate:.1%} (source: {source})")
        else:
            print(f"  - {s['area']}: No learning data yet")
    
    # Check learning status
    status = advisor._get_learning_status(user_id=user_id)
    print(f"\nLearning Status:")
    print(f"  User ID: {status['user_id']}")
    print(f"  Has user-specific data: {status['has_user_specific_data']}")
    print(f"  Has global data: {status['has_global_data']}")
    print(f"  Total learning instances: {status['total_learning_instances']}")

