#!/bin/bash
#
# Launch CORAL on Ursa: Ollama on GPU, agent on service node.
#
# Usage:
#   bash slurm/start_all.sh
#
# After jobs start, connect from your laptop:
#   ssh -L 7860:<SERVICE_NODE>:7860 $USER@ursa-bastion
#   Open http://localhost:7860

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p logs

# Clean up stale host file
HOST_FILE=/scratch5/purged/$USER/coral_host.env
rm -f "$HOST_FILE"

# Submit Ollama GPU job
OLLAMA_JOB=$(sbatch --parsable "$SCRIPT_DIR/start_ollama.sh")
echo "Submitted Ollama job: $OLLAMA_JOB"

# Submit CORAL agent job (starts after Ollama)
CORAL_JOB=$(sbatch --parsable --dependency=after:$OLLAMA_JOB "$SCRIPT_DIR/start_coral.sh")
echo "Submitted CORAL job:  $CORAL_JOB"

echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "After both jobs are RUNNING, connect with:"
echo "  ssh -L 7860:\$(squeue -j $CORAL_JOB -o %N -h):7860 \$USER@ursa-bastion"
echo "  Open http://localhost:7860"
