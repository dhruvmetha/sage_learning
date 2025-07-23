#!/bin/bash

# Script to test the evaluation pipeline on arrakis.cs.rutgers.edu

echo "🚀 Testing Auto Evaluation Pipeline on Arrakis"
echo "============================================="

# Check if we're on arrakis
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" != "arrakis.cs.rutgers.edu" ]]; then
    echo "❌ This script should be run on arrakis.cs.rutgers.edu"
    echo "   Current host: $HOSTNAME"
    echo "   Please SSH to arrakis and run this script there."
    exit 1
fi

# Check conda environment
if [[ "$CONDA_DEFAULT_ENV" != "mjxrl" ]]; then
    echo "⚠️  Expected conda environment 'mjxrl', found '$CONDA_DEFAULT_ENV'"
    echo "   Activating mjxrl environment..."
    source /common/users/dm1487/envs/mjxrl/bin/activate
fi

echo "✅ Running on: $HOSTNAME"
echo "✅ Conda environment: $CONDA_DEFAULT_ENV"
echo "✅ Python path: $(which python)"

# Check GPUs
echo ""
echo "🔍 Checking available GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Test pipeline components
echo ""
echo "🧪 Testing pipeline components..."
python test_pipeline.py

# Show usage instructions
echo ""
echo "📋 Ready to run evaluation pipeline!"
echo ""
echo "Quick test with manual coordination:"
echo "  python src/auto_eval.py coordination=manual max_models_to_detect=1"
echo ""
echo "Full automatic evaluation:"
echo "  python src/auto_eval.py auto_detect_latest=true max_models_to_detect=2"
echo ""
echo "Evaluate specific models:"
echo "  python src/auto_eval.py model_runs=['outputs/2025-07-17/00-16-11']"