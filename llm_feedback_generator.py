"""
LLM-Based Feedback Generator for Golf Swing Improvement

Uses an open-source LLM to generate personalized feedback text.
Falls back to pre-written database if LLM is unavailable.
"""

import numpy as np
from typing import Dict, List, Optional
import json
from swing_improvement_system import SwingImprovementAdvisor


class LLMFeedbackGenerator:
    """
    Generates personalized feedback using an open-source LLM.
    
    Supports multiple LLM backends:
    - Ollama (local, free)
    - Hugging Face Transformers (local)
    - OpenAI-compatible APIs (local servers)
    """
    
    def __init__(
        self,
        llm_backend: str = 'ollama',  # 'ollama', 'huggingface', 'openai_compatible'
        model_name: str = 'llama3.2:1b',  # Model name for the backend
        use_llm: bool = True,
        fallback_to_database: bool = True
    ):
        """
        Args:
            llm_backend: Which LLM backend to use
            model_name: Model name (e.g., 'llama3.2', 'mistral', 'phi-3')
            use_llm: If True, use LLM; if False, use database only
            fallback_to_database: If True, fall back to database if LLM fails
        """
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.use_llm = use_llm
        self.fallback_to_database = fallback_to_database
        self.llm_client = None
        
        # Initialize LLM client if using LLM
        if self.use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM client based on backend"""
        try:
            if self.llm_backend == 'ollama':
                import ollama
                self.llm_client = ollama
                # Test connection
                try:
                    models = ollama.list()  # Check if Ollama is running
                    available_models = [m['name'] for m in models.get('models', [])]
                    print(f"✅ Connected to Ollama")
                    print(f"   Available models: {', '.join(available_models)}")
                    
                    # Check if requested model is available (handle tag format like 'llama3.2:1b')
                    model_found = False
                    for model in available_models:
                        # Check exact match or if model name starts with requested name
                        if model == self.model_name or model.startswith(self.model_name.split(':')[0]):
                            if ':' in self.model_name:
                                # If user specified tag, try to match it
                                if ':' in model and model.split(':')[1] == self.model_name.split(':')[1]:
                                    self.model_name = model
                                    model_found = True
                                    break
                            else:
                                # No tag specified, use first match
                                self.model_name = model
                                model_found = True
                                break
                    
                    if not model_found:
                        print(f"⚠️  Model '{self.model_name}' not found in available models")
                        if available_models:
                            print(f"   Using first available model: {available_models[0]}")
                            self.model_name = available_models[0]
                        else:
                            raise ValueError("No models available in Ollama")
                    
                    print(f"   Using model: {self.model_name}")
                except Exception as e:
                    print(f"⚠️  Ollama not available: {e}")
                    if self.fallback_to_database:
                        print("   Falling back to database")
                        self.use_llm = False
                    else:
                        raise
            
            elif self.llm_backend == 'huggingface':
                from transformers import pipeline
                self.llm_client = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device_map="auto"
                )
                print(f"✅ Loaded Hugging Face model: {self.model_name}")
            
            elif self.llm_backend == 'openai_compatible':
                # For OpenAI-compatible APIs (like local servers)
                import openai
                self.llm_client = openai
                # Configure base URL for local server
                self.llm_client.api_base = "http://localhost:1234/v1"  # Adjust as needed
                self.llm_client.api_key = "not-needed"  # For local servers
                print(f"✅ Configured OpenAI-compatible API")
            
        except ImportError as e:
            print(f"⚠️  LLM library not installed: {e}")
            if self.fallback_to_database:
                print("   Falling back to database")
                self.use_llm = False
            else:
                raise
        except Exception as e:
            print(f"⚠️  LLM initialization failed: {e}")
            if self.fallback_to_database:
                print("   Falling back to database")
                self.use_llm = False
            else:
                raise
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate text using Ollama"""
        # Use the chat API (more reliable)
        response = self.llm_client.chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a professional golf instructor providing helpful, clear feedback.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.7,
                'top_p': 0.9,
                'num_predict': 500
            }
        )
        return response['message']['content'].strip()
    
    def _generate_with_huggingface(self, prompt: str) -> str:
        """Generate text using Hugging Face Transformers"""
        result = self.llm_client(
            prompt,
            max_length=len(prompt.split()) + 200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        generated_text = result[0]['generated_text']
        # Extract only the new text (after prompt)
        new_text = generated_text[len(prompt):].strip()
        return new_text
    
    def _generate_with_openai_compatible(self, prompt: str) -> str:
        """Generate text using OpenAI-compatible API"""
        response = self.llm_client.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional golf instructor providing helpful, clear feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    
    def generate_feedback(
        self,
        area: str,
        issue: str,
        current_score: float,
        target_score: float,
        gap: float,
        impact_level: str,
        swing_type: Optional[str] = None,
        golfer_context: Optional[Dict] = None,
        all_quality_scores: Optional[np.ndarray] = None,
        all_weaknesses: Optional[List[Dict]] = None,
        all_strengths: Optional[List[Dict]] = None
    ) -> Dict[str, str]:
        """
        Generate personalized feedback text using LLM.
        
        Args:
            area: Area of focus (e.g., 'weight_shift')
            issue: Description of the issue
            current_score: Current quality score
            target_score: Target quality score
            gap: Improvement gap
            impact_level: Impact level (minor/moderate/major)
            swing_type: Optional swing type classification
            golfer_context: Optional golfer context (skill level, etc.)
            all_quality_scores: All 8 quality scores (full swing picture)
            all_weaknesses: List of all weaknesses (other issues)
            all_strengths: List of all strengths (what's working well)
        
        Returns:
            Dictionary with 'suggestion', 'drill', 'key_points'
        """
        # Build prompt with full context
        prompt = self._build_prompt(
            area, issue, current_score, target_score, gap,
            impact_level, swing_type, golfer_context,
            all_quality_scores, all_weaknesses, all_strengths
        )
        
        # Generate with LLM if available
        if self.use_llm and self.llm_client:
            try:
                # Generate structured feedback with three components
                structured_prompt = f"""{prompt}

IMPORTANT: Provide feedback in this EXACT structure:

1. WHAT SHOULD HAPPEN (Ideal/Expected):
   [Describe what proper technique looks like for this area - what should be happening in an ideal swing]

2. WHAT IS HAPPENING (Current Issue):
   [Describe what is actually happening in this golfer's swing based on the score and context - be specific about the problem]

3. WHAT TO DO (Actionable Steps):
   [Provide specific, actionable steps the golfer can take to improve - be concrete and practical]

Format your response exactly as shown above with clear section headers."""
                
                structured_feedback = self._generate_text(structured_prompt)
                
                # Parse the structured feedback
                parsed = self._parse_structured_feedback(structured_feedback)
                
                # Generate drill
                drill_prompt = f"{prompt}\n\nProvide a specific practice drill for this issue. Keep it concise and actionable."
                drill = self._generate_text(drill_prompt)
                
                # Generate key points
                key_points_prompt = f"{prompt}\n\nProvide 3-4 key points as a bulleted list. Keep each point short (5-10 words)."
                key_points_text = self._generate_text(key_points_prompt)
                # Parse key points from text
                key_points = self._parse_key_points(key_points_text)
                
                return {
                    'what_should_happen': parsed.get('what_should_happen', ''),
                    'what_is_happening': parsed.get('what_is_happening', ''),
                    'what_to_do': parsed.get('what_to_do', ''),
                    'suggestion': structured_feedback,  # Keep full text for backward compatibility
                    'drill': drill,
                    'key_points': key_points,
                    'generated_by': 'llm'
                }
            
            except Exception as e:
                print(f"⚠️  LLM generation failed: {e}")
                if self.fallback_to_database:
                    return self._get_fallback_feedback(area, issue, impact_level)
                else:
                    raise
        
        # Fallback to database
        if self.fallback_to_database:
            return self._get_fallback_feedback(area, issue, impact_level)
        
        raise RuntimeError("LLM generation failed and fallback disabled")
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the configured LLM backend"""
        if self.llm_backend == 'ollama':
            return self._generate_with_ollama(prompt)
        elif self.llm_backend == 'huggingface':
            return self._generate_with_huggingface(prompt)
        elif self.llm_backend == 'openai_compatible':
            return self._generate_with_openai_compatible(prompt)
        else:
            raise ValueError(f"Unknown LLM backend: {self.llm_backend}")
    
    def _build_prompt(
        self,
        area: str,
        issue: str,
        current_score: float,
        target_score: float,
        gap: float,
        impact_level: str,
        swing_type: Optional[str] = None,
        golfer_context: Optional[Dict] = None,
        all_quality_scores: Optional[np.ndarray] = None,
        all_weaknesses: Optional[List[Dict]] = None,
        all_strengths: Optional[List[Dict]] = None
    ) -> str:
        """Build prompt for LLM with full swing context"""
        
        score_names = [
            'Setup Quality', 'Tempo & Rhythm', 'Weight Shift', 'Body Rotation',
            'Impact Quality', 'Follow-Through', 'Balance', 'Consistency'
        ]
        
        # Base instruction
        prompt = f"""You are a professional golf instructor providing personalized feedback.

Golf Instruction Principles (based on Golf Digest, MyTPI, Hackmotion):
- Focus on optimizing the existing swing, not overhauling it
- Provide minor tweaks for non-critical issues
- Only suggest major changes for critical problems
- Respect individual swing styles
- Be specific and actionable
- Consider how multiple issues interact
- Build on strengths while addressing weaknesses

FULL SWING ANALYSIS:"""
        
        # Add all quality scores (full picture)
        if all_quality_scores is not None:
            prompt += "\n\nAll Quality Scores (0-1, higher is better):"
            for i, (name, score) in enumerate(zip(score_names, all_quality_scores)):
                status = "✓ Good" if score >= 0.70 else "✗ Needs Work"
                prompt += f"\n- {name}: {score:.1%} {status}"
        
        # Add all weaknesses (other issues)
        if all_weaknesses and len(all_weaknesses) > 0:
            prompt += "\n\nAll Areas Needing Improvement:"
            for weakness in all_weaknesses:
                weak_area = weakness['area'].replace('_', ' ').title()
                weak_score = weakness['score']
                weak_gap = weakness.get('gap', 0)
                severity = "Critical" if weak_score < 0.40 else "Moderate" if weak_gap > 0.20 else "Minor"
                
                if weakness['area'] == area:
                    prompt += f"\n- {weak_area}: {weak_score:.1%} ({severity}) ← FOCUS AREA"
                else:
                    prompt += f"\n- {weak_area}: {weak_score:.1%} ({severity})"
        
        # Add strengths (what's working)
        if all_strengths and len(all_strengths) > 0:
            prompt += "\n\nAreas Working Well (build on these):"
            for strength in all_strengths[:3]:  # Top 3 strengths
                strong_area = strength['area'].replace('_', ' ').title()
                strong_score = strength['score']
                prompt += f"\n- {strong_area}: {strong_score:.1%}"
        
        # Current focus area details
        prompt += f"""

CURRENT FOCUS AREA:
- Area: {area.replace('_', ' ').title()}
- Issue: {issue}
- Current Score: {current_score:.1%}
- Target Score: {target_score:.1%}
- Improvement Gap: {gap:.1%}
- Impact Level: {impact_level}"""
        
        # Add issue severity context
        if current_score < 0.40:
            severity_note = "CRITICAL - This is a major issue requiring significant attention."
        elif gap > 0.30:
            severity_note = "SIGNIFICANT - Large improvement needed."
        elif gap > 0.15:
            severity_note = "MODERATE - Noticeable improvement needed."
        else:
            severity_note = "MINOR - Small refinement needed."
        
        prompt += f"\n- Severity: {severity_note}"
        
        # Add specific issue context for weight shift
        if area == 'weight_shift':
            if current_score < 0.50:
                prompt += "\n- Specific Issue: Weight is likely stuck on back foot, not transferring forward during downswing."
            elif current_score < 0.60:
                prompt += "\n- Specific Issue: Weight transfer is happening but too late or insufficient."
            else:
                prompt += "\n- Specific Issue: Weight shift timing or efficiency needs refinement."
        
        # Add context about multiple issues
        if all_weaknesses and len(all_weaknesses) > 1:
            other_issues = [w['area'].replace('_', ' ').title() for w in all_weaknesses if w['area'] != area]
            if other_issues:
                prompt += f"\n- Related Issues: This golfer also struggles with {', '.join(other_issues[:3])}."
                prompt += " Consider how these issues might be connected."
        
        if swing_type:
            prompt += f"\n- Swing Type: {swing_type.replace('_', ' ').title()}"
        
        if golfer_context:
            prompt += f"\n- Golfer Context: {json.dumps(golfer_context, indent=2)}"
        
        prompt += f"""

Instructions:
- Provide clear, actionable feedback for the {area.replace('_', ' ').title()} issue
- Consider how this issue relates to other weaknesses (if any)
- Build on the golfer's strengths
- Focus on {impact_level} changes (not major overhaul unless critical)
- Be encouraging and specific
- Reference professional golf instruction principles
- Be specific about what should happen (ideal technique)
- Be specific about what is actually happening (current problem)
- Provide concrete, actionable steps the golfer can take
- Address the specific type of {area.replace('_', ' ')} problem indicated

Generate personalized feedback for this golfer:"""
        
        return prompt
    
    def _parse_structured_feedback(self, text: str) -> Dict[str, str]:
        """Parse structured feedback into three components"""
        result = {
            'what_should_happen': '',
            'what_is_happening': '',
            'what_to_do': ''
        }
        
        # Try to extract sections by looking for headers
        lines = text.split('\n')
        current_section = None
        current_text = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if 'what should happen' in line_lower or 'ideal' in line_lower or 'expected' in line_lower:
                if current_section and current_text:
                    result[current_section] = ' '.join(current_text).strip()
                current_section = 'what_should_happen'
                current_text = []
                continue
            elif 'what is happening' in line_lower or 'current issue' in line_lower or 'actually happening' in line_lower:
                if current_section and current_text:
                    result[current_section] = ' '.join(current_text).strip()
                current_section = 'what_is_happening'
                current_text = []
                continue
            elif 'what to do' in line_lower or 'actionable' in line_lower or 'steps' in line_lower:
                if current_section and current_text:
                    result[current_section] = ' '.join(current_text).strip()
                current_section = 'what_to_do'
                current_text = []
                continue
            elif line.strip().startswith(('1.', '2.', '3.', '-', '•', '*')):
                # Skip list markers
                line = line.strip()
                for marker in ['1.', '2.', '3.', '-', '•', '*']:
                    if line.startswith(marker):
                        line = line[len(marker):].strip()
                        break
                if line:
                    current_text.append(line)
            elif line.strip() and not line.strip().startswith('[') and not line.strip().endswith(']'):
                # Regular content line
                if current_section:
                    current_text.append(line.strip())
        
        # Save last section
        if current_section and current_text:
            result[current_section] = ' '.join(current_text).strip()
        
        # Fallback: if parsing failed, try to split by numbered sections
        if not any(result.values()):
            # Try alternative parsing
            sections = text.split('\n\n')
            if len(sections) >= 3:
                result['what_should_happen'] = sections[0].strip()
                result['what_is_happening'] = sections[1].strip()
                result['what_to_do'] = sections[2].strip()
        
        # Ensure we have something in each field
        if not result['what_should_happen']:
            result['what_should_happen'] = 'Proper technique should be demonstrated in this area.'
        if not result['what_is_happening']:
            result['what_is_happening'] = 'Current swing shows room for improvement in this area.'
        if not result['what_to_do']:
            result['what_to_do'] = 'Focus on practicing the fundamentals and making gradual improvements.'
        
        return result
    
    def _parse_key_points(self, text: str) -> List[str]:
        """Parse key points from LLM-generated text"""
        # Try to extract bullet points
        lines = text.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            # Remove bullet markers
            for marker in ['-', '•', '*', '1.', '2.', '3.', '4.']:
                if line.startswith(marker):
                    line = line[len(marker):].strip()
                    break
            
            if line and len(line) > 5:  # Valid key point
                key_points.append(line)
        
        # Limit to 4 key points
        return key_points[:4] if key_points else ['Focus on improvement', 'Practice consistently', 'Be patient']
    
    def _get_fallback_feedback(self, area: str, issue: str, impact_level: str) -> Dict[str, str]:
        """Get feedback from database as fallback"""
        advisor = SwingImprovementAdvisor()
        
        # Create dummy scores to trigger suggestion lookup
        score_names = [
            'setup_quality', 'tempo_rhythm', 'weight_shift', 'body_rotation',
            'impact_quality', 'followthrough', 'balance', 'consistency'
        ]
        
        # Generate structured feedback components
        what_should_happen = self._get_ideal_behavior(area)
        what_is_happening = f"Your {area.replace('_', ' ')} shows {issue.lower()}. This is affecting your swing quality."
        what_to_do = ""
        
        if area in score_names:
            # Create scores with this area as weak
            dummy_scores = np.ones(8) * 0.75  # All good
            area_idx = score_names.index(area)
            dummy_scores[area_idx] = 0.50  # Make this area weak
            
            suggestions = advisor.get_improvement_suggestions(dummy_scores, max_suggestions=1)
            if suggestions:
                suggestion = suggestions[0]
                what_to_do = suggestion.get('suggestion', 'Focus on improvement.')
                drill = suggestion.get('drill', 'Practice consistently.')
                key_points = suggestion.get('key_points', ['Focus on improvement'])
                
                return {
                    'what_should_happen': what_should_happen,
                    'what_is_happening': what_is_happening,
                    'what_to_do': what_to_do,
                    'suggestion': f"{what_should_happen}\n\n{what_is_happening}\n\n{what_to_do}",  # Combined for backward compatibility
                    'drill': drill,
                    'key_points': key_points,
                    'generated_by': 'database'
                }
        
        # Default fallback
        what_to_do = f'Focus on improving {area.replace("_", " ")}. {issue}'
        return {
            'what_should_happen': what_should_happen,
            'what_is_happening': what_is_happening,
            'what_to_do': what_to_do,
            'suggestion': f"{what_should_happen}\n\n{what_is_happening}\n\n{what_to_do}",
            'drill': 'Practice this area consistently.',
            'key_points': ['Focus on improvement', 'Practice consistently', 'Be patient'],
            'generated_by': 'database'
        }
    
    def _get_ideal_behavior(self, area: str) -> str:
        """Get description of ideal behavior for an area"""
        ideal_behaviors = {
            'setup_quality': 'In a proper setup, you should have an athletic stance with balanced weight distribution, knees slightly bent, spine angled forward (S-Posture), and consistent alignment to your target. This creates a solid foundation for your swing.',
            'tempo_rhythm': 'Proper tempo involves a smooth, unhurried transition from backswing to downswing. The backswing should set up the downswing, with a natural rhythm that feels controlled and balanced throughout.',
            'weight_shift': 'During the swing, weight should shift from your trail foot in the backswing to your lead foot in the downswing. At impact, your lower body should be ahead of your upper body, with most weight on your lead foot.',
            'body_rotation': 'Your body should rotate smoothly, with your shoulders turning fully in the backswing and your hips leading the downswing. This creates power and consistency through proper sequencing.',
            'impact_quality': 'At impact, the clubface should be square to the target, with your hands slightly ahead of the ball. You should make contact with the ball before the turf, transferring maximum energy to the ball.',
            'followthrough': 'A complete follow-through involves full extension of your arms, weight fully transferred to your lead foot, and your belt buckle facing the target. This indicates proper swing completion and balance.',
            'balance': 'Throughout the swing, you should maintain balance. At the finish, you should be able to hold your position for several seconds, demonstrating control and stability.',
            'consistency': 'A consistent swing repeats the same motion and produces similar results. This comes from proper fundamentals, good tempo, and maintaining balance throughout the swing.'
        }
        return ideal_behaviors.get(area, f'Proper technique in {area.replace("_", " ")} should demonstrate good fundamentals and consistency.')


class LLMEnhancedSwingAdvisor(SwingImprovementAdvisor):
    """
    Enhanced swing advisor that uses LLM for personalized feedback.
    Falls back to database if LLM is unavailable.
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        llm_backend: str = 'ollama',
        llm_model: str = 'llama3.2',
        use_rag: bool = False,  # New: Enable RAG
        knowledge_base_path: str = 'knowledge/golf_instruction/',
        **kwargs
    ):
        """
        Args:
            use_llm: If True, use LLM for feedback generation
            llm_backend: LLM backend ('ollama', 'huggingface', 'openai_compatible')
            llm_model: Model name
            use_rag: If True, use RAG-enhanced feedback (requires rag_feedback_generator)
            knowledge_base_path: Path to golf instruction knowledge base
            **kwargs: Additional arguments for SwingImprovementAdvisor
        """
        super().__init__(**kwargs)
        
        # Use RAG generator if requested, otherwise use base LLM generator
        if use_rag:
            try:
                from rag_feedback_generator import RAGFeedbackGenerator
                self.llm_generator = RAGFeedbackGenerator(
                    llm_backend=llm_backend,
                    model_name=llm_model,
                    use_llm=use_llm,
                    fallback_to_database=True,
                    knowledge_base_path=knowledge_base_path,
                    use_rag=True
                )
                print("✅ Using RAG-enhanced feedback generator")
            except ImportError:
                print("⚠️  RAG not available, falling back to base LLM generator")
                self.llm_generator = LLMFeedbackGenerator(
                    llm_backend=llm_backend,
                    model_name=llm_model,
                    use_llm=use_llm,
                    fallback_to_database=True
                )
        else:
            self.llm_generator = LLMFeedbackGenerator(
                llm_backend=llm_backend,
                model_name=llm_model,
                use_llm=use_llm,
                fallback_to_database=True
            )
    
    def get_improvement_suggestions(
        self,
        quality_scores: np.ndarray,
        max_suggestions: int = 5,
        prioritize: bool = True,
        golfer_context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get suggestions with LLM-generated personalized feedback.
        
        Args:
            quality_scores: 8-dimensional quality array
            max_suggestions: Maximum number of suggestions
            prioritize: If True, prioritize suggestions
            golfer_context: Optional golfer context (skill_level, etc.)
        """
        # Get base suggestions (structure from database)
        base_suggestions = super().get_improvement_suggestions(
            quality_scores, max_suggestions * 2, prioritize
        )
        
        # Get full swing analysis (for context)
        analysis = self.analyze_swing(quality_scores)
        
        # Classify swing type (if using adaptive system)
        swing_type = None
        if hasattr(self, '_classify_swing_type'):
            swing_type = self._classify_swing_type(quality_scores)
        
        # Enhance with LLM-generated text
        enhanced_suggestions = []
        for suggestion in base_suggestions[:max_suggestions]:
            # Generate personalized feedback with full context
            llm_feedback = self.llm_generator.generate_feedback(
                area=suggestion['area'],
                issue=suggestion.get('issue', ''),
                current_score=suggestion.get('current_score', 0.5),
                target_score=suggestion.get('target_score', 0.7),
                gap=suggestion.get('improvement_gap', 0.2),
                impact_level=suggestion.get('impact_level', 'moderate'),
                swing_type=swing_type,
                golfer_context=golfer_context,
                all_quality_scores=quality_scores,  # Full swing picture
                all_weaknesses=analysis['weaknesses'],  # All issues
                all_strengths=analysis['strengths']  # What's working
            )
            
            # Merge LLM feedback with base suggestion
            enhanced = suggestion.copy()
            enhanced['suggestion'] = llm_feedback['suggestion']
            enhanced['drill'] = llm_feedback['drill']
            enhanced['key_points'] = llm_feedback['key_points']
            enhanced['generated_by'] = llm_feedback.get('generated_by', 'database')
            
            enhanced_suggestions.append(enhanced)
        
        return enhanced_suggestions


if __name__ == '__main__':
    # Example usage
    import numpy as np
    
    # Initialize advisor with LLM
    advisor = LLMEnhancedSwingAdvisor(
        use_llm=True,
        llm_backend='ollama',  # or 'huggingface', 'openai_compatible'
        llm_model='llama3.2:1b'  # or 'mistral', 'phi-3', etc.
    )
    
    # Example scores
    scores = np.array([0.65, 0.55, 0.50, 0.60, 0.55, 0.50, 0.70, 0.45])
    
    # Get personalized suggestions
    suggestions = advisor.get_improvement_suggestions(
        scores,
        golfer_context={'skill_level': 'intermediate', 'age': 35}
    )
    
    # Print suggestions
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['area']}")
        print(f"   Generated by: {suggestion.get('generated_by', 'database')}")
        print(f"   Suggestion: {suggestion['suggestion']}")
        print(f"   Drill: {suggestion['drill']}")
        print(f"   Key Points: {', '.join(suggestion['key_points'])}")

