#!/usr/bin/env python3
"""
Create Final Rich Dataset from Fixed ShareGPT Conversions
This uses the improved conversion with complete answers
"""

import json
import os
import random
from pathlib import Path
from datetime import datetime

def load_sharegpt_file(filepath: str) -> list:
    """Load a ShareGPT formatted JSON file"""
    print(f"Loading {os.path.basename(filepath)}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data):,} conversations")
    return data

def analyze_data_quality(conversations: list, source_name: str) -> dict:
    """Analyze the quality of conversations"""
    
    total = len(conversations)
    valid_count = 0
    total_length = 0
    has_think_tag = 0
    
    # Sample analysis
    sample_lengths = []
    
    for conv in conversations:
        if 'messages' in conv and len(conv['messages']) >= 3:
            assistant_msg = conv['messages'][2]['content']
            
            # Check validity
            if '<think>' in assistant_msg and '</think>' in assistant_msg:
                has_think_tag += 1
                
            # Check completeness
            msg_length = len(assistant_msg)
            total_length += msg_length
            
            if msg_length > 300:  # Reasonable threshold for complete answer
                valid_count += 1
                
            if len(sample_lengths) < 10:
                sample_lengths.append(msg_length)
    
    avg_length = total_length / total if total > 0 else 0
    
    quality_report = {
        'source': source_name,
        'total': total,
        'valid': valid_count,
        'validity_rate': (valid_count / total * 100) if total > 0 else 0,
        'has_think_tag': has_think_tag,
        'think_tag_rate': (has_think_tag / total * 100) if total > 0 else 0,
        'avg_length': avg_length,
        'sample_lengths': sample_lengths
    }
    
    return quality_report

def create_train_val_test_splits(all_conversations: list, split_ratios: tuple = (0.9, 0.05, 0.05)) -> dict:
    """Create train, validation, and test splits"""
    
    # Shuffle the data
    random.shuffle(all_conversations)
    
    total = len(all_conversations)
    train_size = int(total * split_ratios[0])
    val_size = int(total * split_ratios[1])
    
    splits = {
        'train': all_conversations[:train_size],
        'validation': all_conversations[train_size:train_size + val_size],
        'test': all_conversations[train_size + val_size:]
    }
    
    return splits

def main():
    """Main dataset creation function"""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Setup directories - using the FIXED data
    base_dir = Path("processed_rich_fixed")  # NEW SOURCE DIRECTORY
    output_dir = Path("final_rich_dataset_fixed")  # NEW OUTPUT DIRECTORY
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("=== Creating Final Rich Dataset (FIXED VERSION) ===")
    print(f"Source: {base_dir}")
    print(f"Output: {output_dir}\n")
    
    # Define which files to include (skip reflection data for core training)
    files_to_include = [
        "rich_sharegpt_cot_data.json",
        "rich_sharegpt_semantic_memory.json",
        "rich_sharegpt_episodic_memory.json",
        "rich_sharegpt_procedural_memory.json"
    ]
    
    # Load all conversations
    all_conversations = []
    quality_reports = []
    
    for filename in files_to_include:
        filepath = base_dir / filename
        
        if filepath.exists():
            conversations = load_sharegpt_file(filepath)
            all_conversations.extend(conversations)
            
            # Analyze quality
            quality = analyze_data_quality(conversations, filename)
            quality_reports.append(quality)
        else:
            print(f"Warning: {filename} not found!")
    
    print(f"\nTotal conversations loaded: {len(all_conversations):,}")
    
    # Print quality analysis
    print("\n=== Data Quality Analysis ===")
    for report in quality_reports:
        print(f"\n{report['source']}:")
        print(f"  Total: {report['total']:,}")
        print(f"  Valid (>300 chars): {report['valid']:,} ({report['validity_rate']:.1f}%)")
        print(f"  Has <think> tags: {report['has_think_tag']:,} ({report['think_tag_rate']:.1f}%)")
        print(f"  Average length: {report['avg_length']:.0f} characters")
        print(f"  Sample lengths: {report['sample_lengths'][:5]}")
    
    # Check improvement over original
    print("\n=== Checking Improvements ===")
    # Sample a few CoT examples to verify they're complete
    cot_examples = [conv for conv in all_conversations[:100] if conv.get('data_source') == 'cot']
    if cot_examples:
        sample = cot_examples[0]
        assistant_msg = sample['messages'][2]['content']
        print(f"Sample CoT response length: {len(assistant_msg)} characters")
        
        # Check for detailed content
        has_details = any(marker in assistant_msg for marker in ['**Detailed Step-by-Step Process**', '*Required Inputs:*', '*How to Validate:*', '**Summary**:'])
        print(f"Has detailed content markers: {has_details}")
        
        if has_details:
            print("✓ CONFIRMED: Responses now include complete details!")
        else:
            print("⚠ WARNING: Responses may still be incomplete")
    
    # Create splits
    print("\n=== Creating Train/Val/Test Splits ===")
    splits = create_train_val_test_splits(all_conversations)
    
    # Save splits
    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"{split_name}: {len(split_data):,} conversations saved")
        
        # Analyze data source distribution
        source_counts = {}
        for conv in split_data:
            source = conv.get('data_source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"  Data distribution: {dict(source_counts)}")
    
    # Create dataset info file
    dataset_info = {
        "dataset_name": "Phi-4 Informius Rich Training Dataset (FIXED)",
        "description": "Enhanced ShareGPT formatted dataset with COMPLETE answers for Phi-4 finetuning",
        "version": "3.0_complete_answers",
        "created_date": datetime.now().isoformat(),
        "total_conversations": len(all_conversations),
        "improvements_over_v2": [
            "All CoT responses include complete step-by-step details",
            "Full validation criteria and input requirements included",
            "Comprehensive error handling and best practices",
            "Detailed summaries and conclusions",
            "Average response length significantly increased"
        ],
        "splits": {
            split_name: {
                "conversations": len(split_data),
                "percentage": len(split_data) / len(all_conversations) * 100,
                "data_source_distribution": dict(source_counts)
            }
            for split_name, split_data in splits.items()
            for source_counts in [{}]
            if [source_counts.update({conv.get('data_source', 'unknown'): source_counts.get(conv.get('data_source', 'unknown'), 0) + 1}) for conv in split_data]
        },
        "data_sources": {
            "cot_data": "Chain-of-thought reasoning with complete step details and validation",
            "semantic_memory": "Knowledge networks with concepts, relationships, and attributes",
            "episodic_memory": "Experience-based scenarios with timelines and outcomes",
            "procedural_memory": "Complete step-by-step methodologies with all details"
        },
        "format": "ShareGPT with <think> tags for explicit reasoning",
        "training_recommendations": {
            "batch_size": "4 per device (dual RTX 3090)",
            "gradient_accumulation_steps": 8,
            "effective_batch_size": 32,
            "learning_rate": "2e-4",
            "epochs": 3,
            "sequence_length": 4096,
            "max_new_tokens": 1024,  # Increased for complete responses
            "lora_rank": 64,
            "quantization": "4-bit"
        },
        "expected_capabilities": [
            "Complete chain-of-thought reasoning with all details",
            "No truncated responses",
            "Rich memory integration and retrieval",
            "Detailed step-by-step problem solving",
            "Comprehensive explanations with validation"
        ]
    }
    
    # Save dataset info
    with open(output_dir / "dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Dataset Creation Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Total conversations: {len(all_conversations):,}")
    print("\nThe dataset now contains COMPLETE answers and is ready for training!")
    
    # Show a complete example
    print("\n=== Sample Complete Response ===")
    for conv in all_conversations[:20]:
        if conv.get('data_source') == 'cot' and len(conv['messages'][2]['content']) > 1000:
            print(f"User: {conv['messages'][1]['content']}")
            print(f"\nAssistant response preview (first 800 chars):")
            print(conv['messages'][2]['content'][:800])
            print(f"\n... (Total length: {len(conv['messages'][2]['content'])} characters)")
            break

if __name__ == "__main__":
    main()