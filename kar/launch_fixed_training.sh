#!/bin/bash
# Phi-4 Informius Training Launch Script - UPDATED VERSION
# Uses the improved dataset with complete answers and proper sequence length

set -e  # Exit on any error

echo "🚀 Phi-4 Informius Training Launcher (UPDATED VERSION)"
echo "===================================================="
echo "Training with COMPLETE ANSWERS dataset and PROPER CONFIGURATION!"
echo ""

# Check GPU availability
echo "📊 Checking GPU setup..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✅ Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 1 ]; then
    echo "❌ No GPUs detected. Training requires at least 1 GPU."
    exit 1
fi

# Display GPU information with memory requirements
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | \
while IFS=, read -r name memory_total memory_free; do
    echo "   GPU: $name (${memory_total}MB total, ${memory_free}MB free)"
done

# Check if we have enough GPU memory
echo ""
echo "⚠️  NOTE: Training with 8192 sequence length requires significant GPU memory."
echo "   Recommended: 24GB+ VRAM per GPU"
echo ""

# Check dataset files
echo "📁 Verifying FIXED dataset files..."
DATASET_DIR="prepared_all_memory_types_corrected"
REQUIRED_FILES=("train.json" "validation.json" "test.json" "dataset_info.json")

if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Fixed dataset directory '$DATASET_DIR' not found!"
    echo "   Please run the data conversion scripts first."
    exit 1
fi

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATASET_DIR/$file" ]; then
        echo "❌ Required file '$DATASET_DIR/$file' not found!"
        exit 1
    else
        FILE_SIZE=$(du -h "$DATASET_DIR/$file" | cut -f1)
        echo "   ✅ $file ($FILE_SIZE)"
    fi
done

# Check config file
if [ ! -f "phi4_axolotl_config_fixed2.yml" ]; then
    echo "❌ Fixed Axolotl config 'phi4_axolotl_config_fixed2.yml' not found!"
    exit 1
fi

# Verify config has correct sequence length
SEQ_LEN=$(grep "sequence_len:" phi4_axolotl_config_fixed2.yml | awk '{print $2}')
if [ "$SEQ_LEN" -lt 8192 ]; then
    echo "⚠️  WARNING: Your config has sequence_len: $SEQ_LEN"
    echo "   This is TOO SHORT for complete answers!"
    echo "   Update to sequence_len: 8192 in your config file"
    read -p "Continue anyway? (not recommended) (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check disk space
echo "💾 Checking disk space..."
FREE_SPACE=$(df . | tail -1 | awk '{print $4}')
FREE_SPACE_GB=$((FREE_SPACE / 1024 / 1024))
echo "   Free space: ${FREE_SPACE_GB}GB"

if [ "$FREE_SPACE_GB" -lt 150 ]; then
    echo "⚠️  Low disk space. Training with longer sequences requires ~150GB for checkpoints."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Show data quality improvements
echo "📈 Dataset Quality Summary:"
echo "   - CoT responses: ~4000 characters (vs ~300 original)"
echo "   - Complete step details included"
echo "   - All validation criteria documented"
echo "   - Comprehensive summaries provided"
echo "   - 1493% increase in content richness!"

# Display UPDATED training configuration
echo "⚙️  UPDATED Training Configuration:"
echo "   Framework: Axolotl"
echo "   Config: phi4_axolotl_config_fixed2.yml"
echo "   Dataset: prepared_all_memory_types_corrected (ALL 7 MEMORY TYPES)"
echo "   Sequence length: 8192 (properly sized for complete responses)"
echo "   Sample packing: DISABLED (to prevent truncation)"
echo "   Micro batch size: 1 (to fit longer sequences)"
echo "   Gradient accumulation: 4 (effective batch size: 4)"
echo "   Max eval tokens: 4096 (for complete evaluation)"
echo "   Train on inputs: TRUE (for better context understanding)"
echo "   Output: phi4_axolotl_outputs_fixed"

# Estimate training time (longer due to increased sequence length and no packing)
TOTAL_EXAMPLES=335900
EFFECTIVE_BATCH_SIZE=4  # micro_batch_size * gradient_accumulation_steps
STEPS_PER_EPOCH=$((TOTAL_EXAMPLES / EFFECTIVE_BATCH_SIZE))
TOTAL_STEPS=$((STEPS_PER_EPOCH * 3))
ESTIMATED_HOURS=$((TOTAL_STEPS / 150))  # Much slower due to longer sequences and no packing

echo "   Estimated training time: ~${ESTIMATED_HOURS} hours (significantly longer due to proper config)"
echo "   Total training steps: ~${TOTAL_STEPS}"

# Memory optimization reminder
echo ""
echo "💡 Memory Optimization Tips:"
echo "   - If OOM occurs, enable DeepSpeed Zero2 or Zero3"
echo "   - Consider reducing sequence_len to 6144 if needed"
echo "   - Flash Attention can help if supported"

# Confirm before starting
echo ""
echo "🎯 Ready to train with PROPERLY CONFIGURED COMPLETE DATASET!"
echo "Command: python -m axolotl.cli.train phi4_axolotl_config_fixed2.yml"
echo ""
echo "⚠️  IMPORTANT: Make sure your config file has:"
echo "   - sequence_len: 8192"
echo "   - sample_packing: false"
echo "   - train_on_inputs: true"
echo "   - eval_max_new_tokens: 4096"
echo ""
read -p "Start training with UPDATED configuration? (Y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Set optimizations
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs if available
export NCCL_P2P_DISABLE=1        # Disable P2P for stability
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Increased for longer sequences

# Additional memory optimizations
export TOKENIZERS_PARALLELISM=false  # Prevent tokenizer warnings

# Create backup of current config
echo "📝 Creating config backup..."
cp phi4_axolotl_config_fixed2.yml "phi4_axolotl_config_backup_$(date +%Y%m%d_%H%M%S).yml"

# Create training session info
echo "📝 Creating training session info..."
SESSION_LOG="training_session_$(date +%Y%m%d_%H%M%S).log"
echo "Training started: $(date)" > $SESSION_LOG
echo "Dataset: prepared_all_memory_types_corrected" >> $SESSION_LOG
echo "Config: phi4_axolotl_config_fixed2.yml" >> $SESSION_LOG
echo "Sequence length: $SEQ_LEN" >> $SESSION_LOG
echo "Expected improvements: Complete answers, no truncation" >> $SESSION_LOG
echo "GPU count: $GPU_COUNT" >> $SESSION_LOG

# Start training
echo "🏁 Starting Phi-4 Axolotl training (UPDATED VERSION)..."
echo "   Time started: $(date)"
echo "   Training log: $SESSION_LOG"
echo ""
echo "Monitor GPU memory usage with: watch -n 1 nvidia-smi"
echo ""

# Run training with nice output
python -m axolotl.cli.train phi4_axolotl_config_fixed2.yml 2>&1 | tee -a $SESSION_LOG

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "🎉 TRAINING COMPLETED SUCCESSFULLY!"
    echo "   Time finished: $(date)"
    echo "   Output directory: phi4_axolotl_outputs_fixed"
    echo "   Your model should now give COMPLETE answers!"
    echo ""
    echo "To test your improved model:"
    echo "   python -m axolotl.cli.inference phi4_axolotl_config_fixed2.yml \\"
    echo "     --max_new_tokens 4096 \\"
    echo "     --temperature 0.7"
    echo ""
    echo "For generation, use these parameters:"
    echo "   max_new_tokens: 4096"
    echo "   temperature: 0.7"
    echo "   do_sample: true"
    echo "   top_p: 0.95"
    echo "   repetition_penalty: 1.1"
else
    echo ""
    echo "❌ Training failed!"
    echo "   Check $SESSION_LOG for details."
    echo ""
    echo "Common issues:"
    echo "   - Out of memory: Try DeepSpeed Zero2/Zero3 or reduce sequence_len"
    echo "   - Config errors: Verify all parameters in phi4_axolotl_config_fixed2.yml"
    echo "   - Data issues: Check dataset format in prepared_all_memory_types_corrected/"
    exit 1
fi