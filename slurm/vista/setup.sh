#!/bin/bash
# CORAL setup script for TACC Vista (Grace Hopper GH200, ARM64)
#
# Usage:
#   source slurm/vista/setup.sh
#
# Prerequisites:
#   - TACC account with active allocation
#   - Internet access on login node

set -euo pipefail

echo "=== CORAL Setup for TACC Vista ==="

# --- Load required modules ---
module load gcc/14.2.0 python3/3.11.8
echo "[1/7] Modules loaded: gcc/14.2.0 python3/3.11.8"

# --- Create install directory ---
INSTALL_DIR=$WORK/coral_install
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"
echo "[2/7] Install directory: $INSTALL_DIR"

# --- Create virtual environment ---
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "export LD_LIBRARY_PATH=/opt/apps/gcc14/cuda12/python3/3.11.8/lib:\$LD_LIBRARY_PATH" >> .venv/bin/activate
    echo "[3/7] Virtual environment created"
else
    echo "[3/7] Virtual environment already exists"
fi
source .venv/bin/activate
pip install --upgrade pip -q

# --- Clone and install CORAL ---
if [ ! -d "coral" ]; then
    git clone https://github.com/mansurjisan/coral.git
    cd coral && git checkout feature/multi-agent && cd ..
    echo "[4/7] CORAL cloned"
else
    cd coral && git pull origin feature/multi-agent && cd ..
    echo "[4/7] CORAL updated"
fi
pip install -e coral -q
pip install pytest pytest-asyncio matplotlib -q

# --- Clone and install ocean-mcp servers ---
if [ ! -d "ocean-mcp" ]; then
    git clone https://github.com/mansurjisan/ocean-mcp.git
    echo "[5/7] ocean-mcp cloned"
else
    cd ocean-mcp && git pull origin main && cd ..
    echo "[5/7] ocean-mcp updated"
fi

cd ocean-mcp/servers
for server in coops-mcp stofs-mcp nhc-mcp recon-mcp erddap-mcp ofs-mcp adcirc-mcp goes-mcp schism-mcp usgs-mcp winds-mcp ww3-mcp ufs-runner-mcp hpc-system-mcp nos-workflow-mcp alert-mcp; do
    if [ -d "$server" ]; then
        cd "$server" && pip install -e . -q 2>/dev/null && cd ..
    fi
done
cd "$INSTALL_DIR"
echo "[6/7] MCP servers installed"

# --- Create Python wrapper for MCP subprocesses ---
cat > "$INSTALL_DIR/python_wrapper.sh" << 'WRAPPER'
#!/bin/bash
export LD_LIBRARY_PATH=/opt/apps/gcc14/cuda12/python3/3.11.8/lib:$LD_LIBRARY_PATH
WRAPPER
echo "exec $INSTALL_DIR/.venv/bin/python \"\$@\"" >> "$INSTALL_DIR/python_wrapper.sh"
chmod +x "$INSTALL_DIR/python_wrapper.sh"

# Update coral_config.json to use the wrapper
cd coral
python3 -c "
import json
with open('coral_config.json') as f:
    cfg = json.load(f)
wrapper = '$INSTALL_DIR/python_wrapper.sh'
for name, srv in cfg['mcpServers'].items():
    srv['command'] = wrapper
with open('coral_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('coral_config.json updated with wrapper path')
"
cd "$INSTALL_DIR"

# --- Install Ollama (ARM64) ---
if [ ! -f "ollama_install/bin/ollama" ]; then
    echo "Downloading Ollama for ARM64..."
    curl -fSL https://github.com/ollama/ollama/releases/download/v0.19.0/ollama-linux-arm64.tar.zst -o ollama.tar.zst
    zstd -d ollama.tar.zst -o ollama.tar
    mkdir -p ollama_install
    tar -xf ollama.tar -C ollama_install
    rm ollama.tar.zst ollama.tar
    echo "[7/7] Ollama installed"
else
    echo "[7/7] Ollama already installed"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start Ollama on a GPU node:"
echo "  sbatch slurm/vista/start_ollama.sh"
echo ""
echo "Then in another terminal:"
echo "  source $INSTALL_DIR/.venv/bin/activate"
echo "  module load gcc/14.2.0 python3/3.11.8"
echo "  export OLLAMA_HOST=http://<gpu-node>:11434"
echo "  cd $INSTALL_DIR/coral"
echo "  coral chat --model qwen3:32b --mode multi"
