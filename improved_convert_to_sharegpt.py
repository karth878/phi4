#!/usr/bin/env python3
"""
Improved ShareGPT Conversion Script
This version includes FULL content in answers, not just placeholders
Output goes to 'processed_rich_fixed/' directory
"""

import json
import os
import random
from typing import Dict, Any, List
from datetime import datetime

def convert_cot_to_sharegpt_rich(item: Dict[str, Any], system_message: str) -> Dict[str, Any]:
    """Convert Chain-of-Thought data with COMPLETE step details in the answer"""
    
    # Extract all available content
    title = item.get('title', 'Chain-of-Thought Reasoning')
    domain = item.get('domain', 'general')
    description = item.get('description', '')
    complexity = item.get('complexity', 'medium')
    
    # Map complexity to descriptions
    complexity_map = {
        'low': 'basic',
        'medium': 'intermediate',
        'high': 'advanced',
        'expert': 'expert: highly complex reasoning with multiple interdependent components'
    }
    complexity_desc = complexity_map.get(complexity, complexity)
    
    # Create varied question templates
    question_templates = [
        f"I need help with {title.lower()} in {domain}. Can you walk me through the systematic approach?",
        f"Walk me through {title.lower()} - I need to understand the reasoning process.",
        f"Please explain the complete methodology for {title.lower()} in {domain} with all steps and details.",
        f"Can you provide a detailed, step-by-step guide for {title.lower()}? Include all validation steps and potential issues.",
        f"I want to master {title.lower()} in {domain}. Give me the comprehensive framework with examples."
    ]
    
    user_question = random.choice(question_templates)
    
    # Build detailed reasoning from actual steps
    reasoning_parts = []
    reasoning_parts.append(f"I need to apply {title} for this {domain} problem at {complexity_desc} level.")
    
    if description:
        reasoning_parts.append(f"This framework: {description}")
    
    # Extract and use actual steps from the pattern
    steps = item.get('steps', [])
    if steps:
        reasoning_parts.append("Let me break this down systematically:")
        
        for i, step in enumerate(steps, 1):
            step_desc = step.get('description', f'Step {i}')
            reasoning_parts.append(f"Step {i}: {step_desc}")
            
            # Add input requirements if available
            inputs = step.get('input_requirements', [])
            if inputs:
                reasoning_parts.append(f"  Required inputs: {', '.join(inputs)}")
            
            # Add validation criteria
            validation = step.get('validation_criteria', [])
            if validation:
                reasoning_parts.append(f"  Validation: {', '.join(validation)}")
            
            # Add potential errors
            errors = step.get('potential_errors', [])
            if errors:
                reasoning_parts.append(f"  Watch out for: {', '.join(errors)}")
    
    # Add prerequisites and key concepts if available
    prerequisites = item.get('prerequisites', [])
    if prerequisites:
        reasoning_parts.append(f"Prerequisites needed: {', '.join(prerequisites)}")
    
    key_concepts = item.get('key_concepts', [])
    if key_concepts:
        reasoning_parts.append(f"Key concepts involved: {', '.join(key_concepts)}")
    
    reasoning_text = "\n".join(reasoning_parts)
    
    # Create COMPREHENSIVE answer with ALL details
    answer_parts = []
    answer_parts.append(f"For {title.lower()} in {domain}, here is the complete systematic approach:")
    
    # Add description if available
    if description:
        answer_parts.append(f"\n**Overview**: {description}")
    
    # Add prerequisites upfront if they exist
    if prerequisites:
        answer_parts.append("\n**Prerequisites**:")
        for prereq in prerequisites:
            answer_parts.append(f"• {prereq}")
    
    # Add ALL steps with FULL details
    if steps:
        answer_parts.append("\n**Detailed Step-by-Step Process**:")
        for i, step in enumerate(steps, 1):
            step_desc = step.get('description', f'Step {i}')
            answer_parts.append(f"\n**Step {i}: {step_desc}**")
            
            # Add ALL input requirements
            inputs = step.get('input_requirements', [])
            if inputs:
                answer_parts.append("*Required Inputs:*")
                for inp in inputs:
                    answer_parts.append(f"  - {inp}")
            
            # Add ALL validation criteria
            validation = step.get('validation_criteria', [])
            if validation:
                answer_parts.append("*How to Validate:*")
                for val in validation:
                    answer_parts.append(f"  - {val}")
            
            # Add ALL potential errors
            errors = step.get('potential_errors', [])
            if errors:
                answer_parts.append("*Common Pitfalls to Avoid:*")
                for err in errors:
                    answer_parts.append(f"  - {err}")
            
            # Add expected outcomes if available
            outcomes = step.get('expected_outcomes', [])
            if outcomes:
                answer_parts.append("*Expected Outcomes:*")
                for outcome in outcomes:
                    answer_parts.append(f"  - {outcome}")
    
    # Add key concepts with explanations
    if key_concepts:
        answer_parts.append("\n**Key Concepts Explained**:")
        for concept in key_concepts:
            answer_parts.append(f"• **{concept}**: Essential for understanding this framework")
    
    # Add best practices if available
    best_practices = item.get('best_practices', [])
    if best_practices:
        answer_parts.append("\n**Best Practices**:")
        for practice in best_practices:
            answer_parts.append(f"• {practice}")
    
    # Add common applications
    applications = item.get('common_applications', [])
    if applications:
        answer_parts.append("\n**Common Applications**:")
        for app in applications:
            answer_parts.append(f"• {app}")
    
    # Add comprehensive conclusion
    answer_parts.append(f"\n**Summary**: This {complexity_desc} framework for {domain} provides a complete, validated methodology for {title.lower()}. By following these detailed steps and validation criteria, you can systematically approach and solve complex problems while avoiding common pitfalls.")
    
    answer = "\n".join(answer_parts)
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": f"<think>\n{reasoning_text}\n</think>\n\n{answer}"}
        ]
    }

def convert_semantic_memory_to_sharegpt_rich(item: Dict[str, Any], system_message: str) -> Dict[str, Any]:
    """Convert semantic memory using actual concept networks"""
    
    # Extract rich semantic content
    memory_content = item.get('memory_content', {})
    concepts = memory_content.get('concepts', [])
    relationships = memory_content.get('relationships', [])
    title = item.get('title', 'Concept Analysis')
    domain = item.get('domain', 'knowledge')
    description = item.get('description', '')
    
    if not concepts:
        # Fallback for different data structures
        concept_data = item.get('concept', {})
        concept_name = concept_data.get('name', 'concept')
        domain = concept_data.get('domain', 'general')
    else:
        # Use first main concept
        main_concept = concepts[0]
        concept_name = main_concept.get('name', main_concept.get('id', 'concept'))
    
    # Create specific questions based on actual content
    question_templates = [
        f"Can you explain {concept_name} and its relationships in {domain}?",
        f"How does {concept_name} connect to other concepts in {domain}?",
        f"What are the key attributes and relationships of {concept_name}?",
        f"Help me understand the concept network around {concept_name} in {domain}.",
        f"Analyze {concept_name} - what should I know about its role in {domain}?"
    ]
    
    user_question = random.choice(question_templates)
    
    # Build reasoning from actual semantic structure
    reasoning_parts = []
    reasoning_parts.append(f"I need to retrieve information about {concept_name} from my semantic memory in the {domain} domain.")
    
    if description:
        reasoning_parts.append(f"Context: {description}")
    
    # Process actual concepts
    if concepts:
        reasoning_parts.append("Let me examine the key concepts in this network:")
        for concept in concepts[:5]:  # Limit to 5 main concepts
            name = concept.get('name', concept.get('id', 'unnamed'))
            definition = concept.get('definition', concept.get('description', ''))
            concept_type = concept.get('type', '')
            
            reasoning_parts.append(f"- {name}: {definition[:100]}{'...' if len(definition) > 100 else ''}")
            if concept_type:
                reasoning_parts.append(f"  Type: {concept_type}")
    
    # Process actual relationships
    if relationships:
        reasoning_parts.append("\nImportant relationships I should consider:")
        for rel in relationships[:4]:  # Limit to 4 key relationships
            source = rel.get('source', '')
            target = rel.get('target', '') 
            rel_type = rel.get('type', rel.get('relation', 'related to'))
            
            reasoning_parts.append(f"- {source} {rel_type} {target}")
            
            # Add confidence if available
            confidence = rel.get('confidence', 0)
            if confidence:
                reasoning_parts.append(f"  (confidence: {confidence}%)")
    
    # Add attributes if available
    attributes = memory_content.get('attributes', [])
    if attributes:
        reasoning_parts.append("\nKey attributes to highlight:")
        for attr in attributes[:3]:
            name = attr.get('name', attr.get('attribute_name', ''))
            value = attr.get('value', attr.get('attribute_value', ''))
            reasoning_parts.append(f"- {name}: {value}")
    
    reasoning_text = "\n".join(reasoning_parts)
    
    # Create detailed answer based on semantic structure
    answer_parts = []
    answer_parts.append(f"**{concept_name}** is a key concept in {domain}.")
    
    if concepts and len(concepts) > 0:
        main_concept = concepts[0]
        definition = main_concept.get('definition', main_concept.get('description', ''))
        if definition:
            answer_parts.append(f"\n**Definition**: {definition}")
    
    if relationships:
        answer_parts.append("\n**Key Relationships**:")
        for rel in relationships[:3]:
            source = rel.get('source', '')
            target = rel.get('target', '')
            rel_type = rel.get('type', rel.get('relation', 'relates to'))
            answer_parts.append(f"• {source} {rel_type} {target}")
    
    if concepts and len(concepts) > 1:
        answer_parts.append("\n**Related Concepts**:")
        for concept in concepts[1:4]:  # Show 3 related concepts
            name = concept.get('name', concept.get('id', ''))
            answer_parts.append(f"• {name}")
    
    answer_parts.append(f"\nThis semantic network in {domain} shows the interconnected nature of knowledge and how {concept_name} fits into the broader conceptual framework.")
    
    answer = "\n".join(answer_parts)
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": f"<think>\n{reasoning_text}\n</think>\n\n{answer}"}
        ]
    }

def convert_episodic_memory_to_sharegpt_rich(item: Dict[str, Any], system_message: str) -> Dict[str, Any]:
    """Convert episodic memory using actual episode content"""
    
    # Extract episode details
    scenario = item.get('scenario', {})
    scenario_desc = scenario.get('description', item.get('scenario_description', 'a research experience'))
    
    context = item.get('context', {})
    timeline = item.get('timeline', [])
    outcome = item.get('outcome', {})
    
    # Create specific questions based on episode content
    question_templates = [
        f"I'm facing a situation similar to {scenario_desc}. Can you share relevant experience?",
        f"Tell me about your experience with {scenario_desc} and what lessons you learned.",
        f"How would you approach {scenario_desc} based on past experience?",
        f"What insights can you share from dealing with {scenario_desc}?",
        f"Walk me through a similar experience to {scenario_desc} and the outcomes."
    ]
    
    user_question = random.choice(question_templates)
    
    # Build reasoning from actual episode
    reasoning_parts = []
    reasoning_parts.append(f"Let me recall my episodic memory related to {scenario_desc}...")
    
    # Add context
    if context:
        setting = context.get('setting', '')
        if setting:
            reasoning_parts.append(f"Setting: {setting}")
    
    # Process timeline events
    if timeline:
        reasoning_parts.append("Key events from this experience:")
        for event in timeline[:5]:
            event_desc = event.get('description', '')
            timestamp = event.get('timestamp', '')
            if event_desc:
                reasoning_parts.append(f"- {event_desc}")
    
    reasoning_text = "\n".join(reasoning_parts)
    
    # Create detailed answer
    answer_parts = []
    answer_parts.append(f"Based on my experience with {scenario_desc}, here's what I learned:")
    
    # Add timeline narrative
    if timeline:
        answer_parts.append("\n**What Happened**:")
        for i, event in enumerate(timeline[:6], 1):
            event_desc = event.get('description', '')
            if event_desc:
                answer_parts.append(f"{i}. {event_desc}")
    
    # Add outcomes
    if outcome:
        success = outcome.get('success', True)
        lessons = outcome.get('lessons_learned', [])
        
        answer_parts.append("\n**Outcome**:")
        answer_parts.append(f"The approach was {'successful' if success else 'challenging'}.")
        
        if lessons:
            answer_parts.append("\n**Key Lessons**:")
            for lesson in lessons[:3]:
                answer_parts.append(f"• {lesson}")
    
    answer_parts.append(f"\nThis experience with {scenario_desc} provides valuable insights for similar situations.")
    
    answer = "\n".join(answer_parts)
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": f"<think>\n{reasoning_text}\n</think>\n\n{answer}"}
        ]
    }

def convert_procedural_memory_to_sharegpt_rich(item: Dict[str, Any], system_message: str) -> Dict[str, Any]:
    """Convert procedural memory with COMPLETE methodology details"""
    
    # Extract procedural content
    memory_content = item.get('memory_content', {})
    steps = memory_content.get('steps', [])
    title = item.get('title', 'Procedural Knowledge')
    domain = item.get('domain', 'general')
    description = item.get('description', '')
    
    procedure_name = title.lower().replace('implementing', '').replace('performing', '').strip()
    
    # Create specific questions
    question_templates = [
        f"How do I {procedure_name} step by step?",
        f"Can you walk me through the complete process of {procedure_name}?",
        f"I need detailed instructions for {procedure_name} in {domain}.",
        f"What's the proper methodology for {procedure_name}? Include all details.",
        f"Explain the complete procedure for {procedure_name} with best practices."
    ]
    
    user_question = random.choice(question_templates)
    
    # Build reasoning
    reasoning_parts = []
    reasoning_parts.append(f"Let me retrieve the procedural knowledge for {procedure_name} in {domain}...")
    
    if description:
        reasoning_parts.append(f"Overview: {description}")
    
    # Add step preview
    if steps:
        reasoning_parts.append(f"This involves {len(steps)} key steps that must be performed in sequence.")
    
    reasoning_text = "\n".join(reasoning_parts)
    
    # Create COMPLETE answer with all procedural details
    answer_parts = []
    answer_parts.append(f"Here's the complete procedure for {procedure_name}:")
    
    if description:
        answer_parts.append(f"\n**Overview**: {description}")
    
    # Add conditions/prerequisites
    conditions = memory_content.get('conditions', [])
    if conditions:
        answer_parts.append("\n**Prerequisites**:")
        for condition in conditions:
            answer_parts.append(f"• {condition}")
    
    # Add ALL steps with complete details
    if steps:
        answer_parts.append("\n**Detailed Step-by-Step Process**:")
        for i, step in enumerate(steps, 1):
            step_desc = step.get('description', step.get('action', f'Step {i}'))
            answer_parts.append(f"\n**Step {i}: {step_desc}**")
            
            # Add step ID if available
            step_id = step.get('step_id', '')
            if step_id:
                answer_parts.append(f"*Step ID: {step_id}*")
            
            # Add expected duration
            duration = step.get('expected_duration', '')
            if duration:
                answer_parts.append(f"*Expected Duration: {duration}*")
            
            # Add specific actions
            action = step.get('action', '')
            if action and action != step_desc:
                answer_parts.append(f"*Action: {action}*")
            
            # Add any additional details
            details = step.get('details', [])
            if details:
                answer_parts.append("*Additional Details:*")
                for detail in details:
                    answer_parts.append(f"  - {detail}")
    
    # Add best practices
    best_practices = memory_content.get('best_practices', [])
    if best_practices:
        answer_parts.append("\n**Best Practices**:")
        for practice in best_practices:
            answer_parts.append(f"• {practice}")
    
    # Add common pitfalls
    pitfalls = memory_content.get('common_pitfalls', [])
    if pitfalls:
        answer_parts.append("\n**Common Pitfalls to Avoid**:")
        for pitfall in pitfalls:
            answer_parts.append(f"• {pitfall}")
    
    # Add comprehensive conclusion
    answer_parts.append(f"\n**Summary**: This complete procedural guide for {procedure_name} ensures systematic and effective execution. Follow each step carefully, paying attention to the prerequisites and best practices for optimal results.")
    
    answer = "\n".join(answer_parts)
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": f"<think>\n{reasoning_text}\n</think>\n\n{answer}"}
        ]
    }

def convert_realtime_reflection_to_sharegpt_rich(item: Dict[str, Any], system_message: str) -> Dict[str, Any]:
    """Convert real-time reflection using actual reflection content"""
    
    scenario_desc = item.get('scenario_description', 'complex problem-solving')
    current_situation = item.get('current_situation', {})
    
    # Create specific questions based on reflection content
    question_templates = [
        f"I'm in the middle of {scenario_desc} and need quick reflection on my approach. Help me adjust in real-time.",
        f"Can you give me immediate feedback on my current strategy for {scenario_desc}?",
        f"I need rapid course correction for {scenario_desc}. What should I adjust right now?",
        f"Quick reflection needed: How should I modify my approach to {scenario_desc}?",
        f"Real-time guidance please - I'm working on {scenario_desc} and need immediate insights."
    ]
    
    user_question = random.choice(question_templates)
    
    # Build quick reflection reasoning (real-time constraint)
    reasoning_parts = []
    reasoning_parts.append(f"Quick real-time reflection on {scenario_desc}...")
    
    # Add current state analysis
    current_state = current_situation.get('current_state', '')
    if current_state:
        reasoning_parts.append(f"Current state: {current_state}")
    
    # Add immediate observations
    observations = item.get('immediate_observations', [])
    if observations:
        reasoning_parts.append("Key observations:")
        for obs in observations[:3]:
            reasoning_parts.append(f"- {obs}")
    
    reasoning_text = "\n".join(reasoning_parts)
    
    # Create rapid reflection answer
    answer_parts = []
    answer_parts.append(f"**Quick reflection on {scenario_desc}:**")
    
    # Add immediate adjustments
    adjustments = item.get('immediate_adjustments', [])
    if adjustments:
        answer_parts.append("\n**Immediate Adjustments Needed**:")
        for adj in adjustments[:3]:
            answer_parts.append(f"• {adj}")
    
    # Add quick wins
    quick_wins = item.get('quick_wins', [])
    if quick_wins:
        answer_parts.append("\n**Quick Wins Available**:")
        for win in quick_wins[:2]:
            answer_parts.append(f"• {win}")
    
    # Add warning signs
    warning_signs = item.get('warning_signs', [])
    if warning_signs:
        answer_parts.append("\n**Watch Out For**:")
        for sign in warning_signs[:2]:
            answer_parts.append(f"• {sign}")
    
    answer_parts.append("\nMake these adjustments now for better outcomes.")
    
    answer = "\n".join(answer_parts)
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": f"<think>\n{reasoning_text}\n</think>\n\n{answer}"}
        ]
    }

def convert_strategy_reflection_to_sharegpt_rich(item: Dict[str, Any], system_message: str) -> Dict[str, Any]:
    """Convert strategy reflection using actual strategic analysis"""
    
    scenario_desc = item.get('scenario_description', 'strategic planning')
    
    # Create strategic questions
    question_templates = [
        f"I need strategic reflection on {scenario_desc}. What patterns and approaches should I consider?",
        f"Can you analyze the strategic landscape for {scenario_desc}?",
        f"Help me develop a strategic approach to {scenario_desc} based on reflection and analysis.",
        f"What strategic insights can you share about {scenario_desc}?",
        f"I'm planning my strategy for {scenario_desc}. What should I consider?"
    ]
    
    user_question = random.choice(question_templates)
    
    # Build strategic reasoning
    reasoning_parts = []
    reasoning_parts.append(f"Strategic reflection on {scenario_desc}...")
    
    # Add strategic context
    context = item.get('strategic_context', {})
    if context:
        timeframe = context.get('timeframe', '')
        if timeframe:
            reasoning_parts.append(f"Timeframe: {timeframe}")
    
    reasoning_text = "\n".join(reasoning_parts)
    
    # Create strategic answer
    answer_parts = []
    answer_parts.append(f"**Strategic reflection on {scenario_desc}:**")
    
    # Add strategic patterns
    patterns = item.get('strategic_patterns', [])
    if patterns:
        answer_parts.append("\n**Key Strategic Patterns**:")
        for pattern in patterns[:3]:
            answer_parts.append(f"• {pattern}")
    
    # Add opportunities
    opportunities = item.get('opportunities', [])
    if opportunities:
        answer_parts.append("\n**Strategic Opportunities**:")
        for opp in opportunities[:3]:
            answer_parts.append(f"• {opp}")
    
    # Add risks
    risks = item.get('risks', [])
    if risks:
        answer_parts.append("\n**Strategic Risks**:")
        for risk in risks[:3]:
            answer_parts.append(f"• {risk}")
    
    # Add recommendations
    recommendations = item.get('strategic_recommendations', [])
    if recommendations:
        answer_parts.append("\n**Strategic Recommendations**:")
        for rec in recommendations[:3]:
            answer_parts.append(f"• {rec}")
    
    answer_parts.append(f"\nThis strategic analysis of {scenario_desc} provides a foundation for informed decision-making.")
    
    answer = "\n".join(answer_parts)
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": f"<think>\n{reasoning_text}\n</think>\n\n{answer}"}
        ]
    }

def convert_deep_reflection_to_sharegpt_rich(item: Dict[str, Any], system_message: str) -> Dict[str, Any]:
    """Convert deep reflection using comprehensive analysis"""
    
    scenario_desc = item.get('scenario_description', 'complex situation')
    
    # Create deep reflection questions
    question_templates = [
        f"I need deep reflection on {scenario_desc}. What are the fundamental patterns and implications?",
        f"Can you provide a comprehensive analysis of {scenario_desc} at a fundamental level?",
        f"Help me understand the deeper implications of {scenario_desc}.",
        f"What systemic insights emerge from deep reflection on {scenario_desc}?",
        f"Analyze {scenario_desc} deeply - what are the root causes and long-term effects?"
    ]
    
    user_question = random.choice(question_templates)
    
    # Build deep reflection reasoning
    reasoning_parts = []
    reasoning_parts.append(f"Deep reflection and analysis of {scenario_desc}...")
    
    # Add fundamental analysis
    fundamentals = item.get('fundamental_analysis', {})
    if fundamentals:
        core_issue = fundamentals.get('core_issue', '')
        if core_issue:
            reasoning_parts.append(f"Core issue: {core_issue}")
    
    reasoning_text = "\n".join(reasoning_parts)
    
    # Create comprehensive deep reflection answer
    answer_parts = []
    answer_parts.append(f"**Deep reflection analysis of {scenario_desc}:**")
    
    # Add fundamental insights
    patterns = item.get('underlying_patterns', [])
    if patterns:
        answer_parts.append("\n**Underlying Patterns**:")
        for pattern in patterns[:3]:
            answer_parts.append(f"• {pattern}")
    
    # Add root causes
    root_causes = item.get('root_causes', [])
    if root_causes:
        answer_parts.append("\n**Root Causes**:")
        for cause in root_causes[:3]:
            answer_parts.append(f"• {cause}")
    
    # Add implications
    implications = item.get('long_term_implications', [])
    if implications:
        answer_parts.append("\n**Long-term Implications**:")
        for implication in implications[:3]:
            answer_parts.append(f"• {implication}")
    
    # Add learning insights
    learning_insights = item.get('learning_insights', [])
    if learning_insights:
        answer_parts.append("\n**Key Learning**:")
        for insight in learning_insights[:2]:
            answer_parts.append(f"• {insight}")
    
    # Add system recommendations
    system_recommendations = item.get('system_level_recommendations', [])
    if system_recommendations:
        answer_parts.append("\n**System-Level Changes**:")
        for rec in system_recommendations[:3]:
            answer_parts.append(f"• {rec}")
    
    answer_parts.append(f"\nThis deep reflection reveals the fundamental dynamics underlying {scenario_desc} and provides a foundation for systemic improvements.")
    
    answer = "\n".join(answer_parts)
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": f"<think>\n{reasoning_text}\n</think>\n\n{answer}"}
        ]
    }

def process_file(input_file: str, data_type: str, output_file: str, max_examples: int = None) -> int:
    """Process a consolidated data file and convert to ShareGPT format using rich content"""
    
    print(f"Processing {data_type} data from {os.path.basename(input_file)}...")
    
    conversations = []
    
    # Define system message
    system_message = "You are Phi, an AI assistant trained to provide detailed reasoning before answering questions. You excel at chain-of-thought reasoning, memory integration, and reflective analysis."
    
    # Define conversion functions
    conversion_functions = {
        "cot": convert_cot_to_sharegpt_rich,
        "semantic_memory": convert_semantic_memory_to_sharegpt_rich,
        "episodic_memory": convert_episodic_memory_to_sharegpt_rich,
        "procedural_memory": convert_procedural_memory_to_sharegpt_rich,
        "realtime_reflection": convert_realtime_reflection_to_sharegpt_rich,
        "strategy_reflection": convert_strategy_reflection_to_sharegpt_rich,
        "deep_reflection": convert_deep_reflection_to_sharegpt_rich
    }
    
    converter = conversion_functions.get(data_type)
    if not converter:
        print(f"Warning: No converter for {data_type}")
        return 0
    
    # Process based on file format
    count = 0
    
    try:
        if input_file.endswith('.json'):
            # Handle JSON format (for CoT data)
            with open(input_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        items = data
                    else:
                        items = [data]
                except json.JSONDecodeError:
                    print(f"Error reading JSON file: {input_file}")
                    return 0
                    
            for item in items:
                if max_examples and count >= max_examples:
                    break
                    
                try:
                    conversation = converter(item, system_message)
                    conversation['data_source'] = data_type
                    conversations.append(conversation)
                    count += 1
                    
                    if count % 1000 == 0:
                        print(f"  Processed {count:,} {data_type} examples...")
                        
                except Exception as e:
                    print(f"  Error processing item: {str(e)[:100]}")
                    continue
                    
        else:
            # Handle JSONL format
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if max_examples and count >= max_examples:
                        break
                        
                    try:
                        item = json.loads(line.strip())
                        conversation = converter(item, system_message)
                        conversation['data_source'] = data_type
                        conversations.append(conversation)
                        count += 1
                        
                        if count % 1000 == 0:
                            print(f"  Processed {count:,} {data_type} examples...")
                            
                    except json.JSONDecodeError:
                        print(f"  Skipping malformed line {line_num + 1}")
                        continue
                    except Exception as e:
                        print(f"  Error on line {line_num + 1}: {str(e)[:100]}")
                        continue
    
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return 0
    
    # Save conversations
    if conversations:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved {count:,} {data_type} conversations to {output_file}")
        
        # Show sample of improved output
        if conversations:
            sample = conversations[0]
            assistant_msg = sample['messages'][2]['content']
            print(f"  Sample output length: {len(assistant_msg)} characters")
            if len(assistant_msg) > 500:
                print("  ✓ Output includes complete details!")
    
    return count

def convert_all_data():
    """Convert all data files to ShareGPT format with complete content"""
    
    # Set up directories - using NEW output directory
    base_dir = "raw_consolidated"
    output_dir = "processed_rich_fixed"  # NEW DIRECTORY
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Converting All Data to Rich ShareGPT Format (FIXED VERSION) ===\n")
    print(f"Output directory: {output_dir}\n")
    
    # File mappings with sample limits for manageable sizes
    file_mappings = [
        ("combined_cot_data.json", "cot", "rich_sharegpt_cot_data.json", 50000),
        ("combined_semantic_memory.jsonl", "semantic_memory", "rich_sharegpt_semantic_memory.json", 30000),
        ("combined_episodic_memory.jsonl", "episodic_memory", "rich_sharegpt_episodic_memory.json", 2000),
        ("combined_procedural_memory.jsonl", "procedural_memory", "rich_sharegpt_procedural_memory.json", 1000),
        ("combined_realtime_reflection.jsonl", "realtime_reflection", "rich_sharegpt_realtime_reflection.json", 20000),
        ("combined_strategy_reflection.jsonl", "strategy_reflection", "rich_sharegpt_strategy_reflection.json", 15000),
        ("combined_deep_reflection.jsonl", "deep_reflection", "rich_sharegpt_deep_reflection.json", 5000)
    ]
    
    total_conversations = 0
    
    for input_filename, data_type, output_filename, max_examples in file_mappings:
        input_file = os.path.join(base_dir, input_filename)
        output_file = os.path.join(output_dir, output_filename)
        
        if os.path.exists(input_file):
            count = process_file(input_file, data_type, output_file, max_examples)
            total_conversations += count
        else:
            print(f"Warning: {input_filename} not found, skipping...")
    
    print(f"\n=== Rich ShareGPT Conversion Complete ===")
    print(f"Total conversations created: {total_conversations:,}")
    print(f"All converted files saved in: {output_dir}")
    print("\nNOTE: This is the FIXED version with complete answers!")
    
    # Create info file
    info = {
        "conversion_date": datetime.now().isoformat(),
        "version": "fixed_complete_answers",
        "improvements": [
            "All CoT steps include full details",
            "Complete validation criteria included",
            "All input requirements listed",
            "Potential errors and pitfalls documented",
            "Best practices and applications added",
            "Comprehensive summaries provided"
        ],
        "total_conversations": total_conversations
    }
    
    with open(os.path.join(output_dir, "conversion_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    return total_conversations

if __name__ == "__main__":
    random.seed(42)  # For reproducible sampling
    convert_all_data()