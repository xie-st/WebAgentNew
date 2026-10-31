#!/bin/bash

################################################################################
# effGen Installation Engine (CANONICAL installer)
#
# This is the one install engine for effGen. It owns the driver-aware PyTorch
# selection (install_compatible_torch below) so a CUDA-13 default wheel is never
# silently installed on a CUDA-12 driver, and handles conda environment
# management, dependency installation, configuration, model downloads, and
# verification. The repo-root ./install.sh is a thin front-end that delegates
# here; scripts/setup_and_verify.sh is a deprecated forwarder. Run either
# ./install.sh (recommended) or this script directly — both do the same work.
#
# Usage:
#   bash scripts/install_effgen.sh [OPTIONS]
#
# Options:
#   --skip-conda           Use system Python instead of conda
#   --skip-verification    Skip verification tests
#   --download-models      Download popular SLM models
#   --install-vllm         Install vLLM for faster inference
#   --dev                  Install development dependencies
#   --minimal              Install only minimal dependencies
#   --help                 Show this help message
#
# Requirements:
#   - conda (recommended) or Python 3.11+
#   - CUDA toolkit (optional, for GPU support)
################################################################################

set -e  # Exit on error

################################################################################
# Colors and Formatting
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

BOLD='\033[1m'
DIM='\033[2m'
UNDERLINE='\033[4m'

################################################################################
# Script Configuration
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check for inherited settings from parent script
QUICK_MODE="${EFFGEN_QUICK_MODE:-false}"

# Animation delay (0 in quick mode)
if [ "$QUICK_MODE" = true ]; then
    ANIM_DELAY=0
else
    ANIM_DELAY=0.1
fi

CONDA_ENV_NAME="effgen"
SKIP_CONDA=false
SKIP_VERIFICATION=false
DOWNLOAD_MODELS=false
INSTALL_VLLM=false
INSTALL_DEV=false
MINIMAL_INSTALL=false

################################################################################
# Helper Functions
################################################################################

# Smooth delay function - skips in quick mode
delay() {
    if [ "$QUICK_MODE" = false ] && [ -n "$1" ]; then
        sleep "$1"
    fi
}

line_delay() { delay "$ANIM_DELAY"; }
section_delay() { delay "$(echo "$ANIM_DELAY * 3" | bc -l 2>/dev/null || echo 0.3)"; }
phase_delay() { delay "$(echo "$ANIM_DELAY * 5" | bc -l 2>/dev/null || echo 0.5)"; }

# Wipe the screen, but only when there is a screen that can be wiped. Without a
# TERM to describe the terminal, `clear` fails with "TERM environment variable
# not set." and, under `set -e`, ends the run — which is what a CI runner, a
# piped install and `nohup` all get.
clear_screen() {
    if [ -t 1 ] && [ -n "${TERM:-}" ] && [ "${TERM}" != "dumb" ]; then
        clear || true
    fi
}

print_banner() {
    if [ "$QUICK_MODE" = false ]; then
        clear_screen
    fi
    echo ""
    section_delay
    echo -e "${CYAN}${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}                                                                ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}              ${MAGENTA}${BOLD}🤖  EFFGEN INSTALLER  🤖${NC}              ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}                                                                ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}        ${WHITE}Complete Setup for Multi-Agent AI Framework${NC}        ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}                                                                ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    line_delay
    echo -e "${YELLOW}${BOLD}✨ Features:${NC}"
    line_delay
    echo -e "  ${GREEN}▸${NC} vLLM & Transformers Support"
    line_delay
    echo -e "  ${GREEN}▸${NC} Multi-GPU Management"
    line_delay
    echo -e "  ${GREEN}▸${NC} Automatic Configuration"
    line_delay
    echo -e "  ${GREEN}▸${NC} Post-Install Verification"
    echo ""
    section_delay
}

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_subheader() {
    echo ""
    echo -e "${MAGENTA}${BOLD}▶ $1${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}${BOLD}✓${NC} ${GREEN}$1${NC}"
}

print_success_big() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "    ╔═══════════════════════════════════════════════════════════╗"
    echo "    ║                                                           ║"
    echo -e "    ║              ${WHITE}🎉  $1  🎉${NC}${GREEN}${BOLD}              ║"
    echo "    ║                                                           ║"
    echo "    ╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

print_error() {
    echo -e "${RED}${BOLD}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}${BOLD}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}${BOLD}ℹ${NC} $1"
}

print_step() {
    echo -e "${WHITE}${BOLD}→${NC} $1"
}

print_progress() {
    echo -e "${CYAN}⏳${NC} $1"
}

print_feature() {
    echo -e "${GREEN}  ●${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_python_version() {
    local version=$($1 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)

    if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
        return 0
    else
        return 1
    fi
}

show_help() {
    head -n 30 "$0" | grep "^#" | sed 's/^# \?//'
    exit 0
}

spinner() {
    local pid=$1
    local message=$2
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0

    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) %10 ))
        printf "\r${CYAN}${spin:$i:1}${NC} $message"
        sleep 0.1
    done
    printf "\r"
}

progress_with_timer() {
    local pid=$1
    local message=$2
    local estimated_time=$3  # in seconds
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    local start_time=$(date +%s)

    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) %10 ))
        local elapsed=$(($(date +%s) - start_time))
        local mins=$((elapsed / 60))
        local secs=$((elapsed % 60))

        # Calculate progress bar
        local progress=0
        if [ $estimated_time -gt 0 ]; then
            progress=$((elapsed * 100 / estimated_time))
            if [ $progress -gt 99 ]; then
                progress=99
            fi
        fi

        # Create progress bar (30 chars wide)
        local filled=$((progress * 30 / 100))
        local empty=$((30 - filled))
        local bar=""
        for ((j=0; j<filled; j++)); do bar="${bar}█"; done
        for ((j=0; j<empty; j++)); do bar="${bar}░"; done

        # Display with spinner, progress bar, and timer
        printf "\r${CYAN}${spin:$i:1}${NC} ${message} ${YELLOW}[${bar}]${NC} ${DIM}%02d:%02d elapsed${NC}" $mins $secs
        sleep 0.1
    done

    # Show final time
    local total_elapsed=$(($(date +%s) - start_time))
    local total_mins=$((total_elapsed / 60))
    local total_secs=$((total_elapsed % 60))
    printf "\r${GREEN}✓${NC} ${message} ${GREEN}[████████████████████████████████]${NC} ${DIM}%02d:%02d${NC}\n" $total_mins $total_secs
}

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            ANIM_DELAY=0
            shift
            ;;
        --skip-conda)
            SKIP_CONDA=true
            shift
            ;;
        --skip-verification)
            SKIP_VERIFICATION=true
            shift
            ;;
        --download-models)
            DOWNLOAD_MODELS=true
            shift
            ;;
        --install-vllm)
            INSTALL_VLLM=true
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --minimal)
            MINIMAL_INSTALL=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Banner and Welcome
################################################################################

if [ "$QUICK_MODE" = false ]; then
    clear_screen
fi
print_banner

echo -e "${WHITE}This installer will set up the complete effGen framework:${NC}"
echo ""
print_feature "Conda environment management (effgen)"
print_feature "All dependencies and requirements"
print_feature "Framework installation (pip install -e .)"
print_feature "Configuration files and API key setup"
print_feature "Optional model downloads"
print_feature "Verification tests"
echo ""

phase_delay

################################################################################
# System Prerequisites Check
################################################################################

print_header "🔍 System Prerequisites Check"

# Check for conda
print_step "Checking for conda installation..."
if [ "$SKIP_CONDA" = false ]; then
    if check_command conda; then
        CONDA_VERSION=$(conda --version)
        print_success "Found: $CONDA_VERSION"
        USE_CONDA=true
    else
        print_warning "Conda not found - will use system Python"
        USE_CONDA=false
        SKIP_CONDA=true
    fi
else
    print_info "Skipping conda (--skip-conda specified)"
    USE_CONDA=false
fi

# Check Python
if [ "$USE_CONDA" = false ]; then
    print_step "Checking Python installation..."
    PYTHON_CMD=""

    for cmd in python3 python; do
        if check_command $cmd; then
            if check_python_version $cmd; then
                PYTHON_CMD=$cmd
                VERSION=$($cmd --version 2>&1)
                print_success "Found: $VERSION"
                break
            fi
        fi
    done

    if [ -z "$PYTHON_CMD" ]; then
        print_error "Python 3.11+ is required but not found"
        print_info "Please install Python 3.11 or higher"
        exit 1
    fi
fi

# Check pip
if [ "$USE_CONDA" = false ]; then
    print_step "Checking pip installation..."
    if $PYTHON_CMD -m pip --version &> /dev/null; then
        PIP_VERSION=$($PYTHON_CMD -m pip --version)
        print_success "Found: $PIP_VERSION"
    else
        print_error "pip is not installed"
        exit 1
    fi
fi

# Check git
print_step "Checking git installation..."
if check_command git; then
    GIT_VERSION=$(git --version)
    print_success "Found: $GIT_VERSION"
else
    print_warning "git not found (optional but recommended)"
fi

# Check CUDA
print_step "Checking for CUDA support..."
if check_command nvidia-smi; then
    # Count GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    print_success "Detected ${GPU_COUNT} NVIDIA GPU(s)"

    echo ""
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║                        GPU INFORMATION                           ║${NC}"
    echo -e "${CYAN}${BOLD}╠════╦═══════════════════════╦═══════════════╦══════════════════╣${NC}"
    echo -e "${CYAN}${BOLD}║ ID ║ Model                 ║ Driver        ║ Memory           ║${NC}"
    echo -e "${CYAN}${BOLD}╠════╬═══════════════════════╬═══════════════╬══════════════════╣${NC}"

    # Collect and display GPU info in table format
    gpu_id=0
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while IFS=',' read -r name driver memory; do
        # Trim whitespace
        name=$(echo "$name" | xargs)
        driver=$(echo "$driver" | xargs)
        memory=$(echo "$memory" | xargs)

        # Format strings to fit table width
        printf "${WHITE}║ %-2s ║ %-21s ║ %-13s ║ %-16s ║${NC}\n" "$gpu_id" "$name" "$driver" "$memory"
        gpu_id=$((gpu_id + 1))
    done

    echo -e "${CYAN}${BOLD}╚════╩═══════════════════════╩═══════════════╩══════════════════╝${NC}"

    HAS_GPU=true

    # Capture the maximum CUDA runtime the installed driver supports (e.g. "12.4").
    # We use this to install a matching PyTorch wheel so a CUDA-13 default wheel is
    # not silently installed on a CUDA-12 driver (which disables the GPU).
    DRIVER_CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [ -n "$DRIVER_CUDA_VERSION" ]; then
        print_info "Driver supports up to CUDA ${DRIVER_CUDA_VERSION}"
    fi
else
    print_info "No NVIDIA GPU detected - will use CPU mode"
    HAS_GPU=false
    DRIVER_CUDA_VERSION=""
fi

echo ""
phase_delay

################################################################################
# Conda Environment Setup
################################################################################

if [ "$USE_CONDA" = true ]; then
    print_header "🐍 Conda Environment Setup"

    # Check if environment already exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_info "Conda environment '${CONDA_ENV_NAME}' already exists"
        # Only ask when there is somebody to answer. Run from CI, from another
        # script, or with stdin redirected, `read` returns non-zero at once and
        # `set -e` would end the install with "Installation failed" — for a
        # condition that is not a failure. With no terminal, keep the existing
        # environment: it is the choice that destroys nothing, and installing
        # into it is what the next step does anyway.
        REPLY=""
        if [ -t 0 ]; then
            read -p "$(echo -e ${YELLOW}Do you want to recreate it? This will delete the existing environment. [y/N]:${NC} )" -n 1 -r || REPLY=""
            echo
        else
            print_info "No terminal to ask on — keeping the existing environment."
            print_info "To rebuild it from scratch, run: conda env remove -n ${CONDA_ENV_NAME}"
        fi
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_progress "Removing existing environment..."
            conda env remove -n "$CONDA_ENV_NAME" -y > /dev/null 2>&1
            print_success "Existing environment removed"
        else
            print_info "Using existing environment"
        fi
    fi

    # Create environment if it doesn't exist
    if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_progress "Creating conda environment '${CONDA_ENV_NAME}' with Python 3.11..."
        conda create -n "$CONDA_ENV_NAME" python=3.11 -y > /dev/null 2>&1
        print_success "Conda environment created"
    fi

    # Activate environment
    print_step "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    print_success "Environment activated: ${CONDA_ENV_NAME}"

    # Set Python command
    PYTHON_CMD="python"

    echo ""
    phase_delay
fi

################################################################################
# Dependency Installation
################################################################################

print_header "📦 Installing Dependencies"

cd "$PROJECT_ROOT"

# Upgrade pip
print_step "Upgrading pip, setuptools, and wheel..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_success "Core tools upgraded"

# Install a PyTorch build that matches the host CUDA driver BEFORE installing the
# rest of the dependencies. Installing torch here pins a driver-compatible wheel so
# the later `pip install -e .` (which only constrains `torch>=2.0,<3`) does not pull
# the default PyPI wheel, which may be built for a newer CUDA runtime than the
# driver supports and would silently fall back to CPU.
install_compatible_torch() {
    if [ "$HAS_GPU" != true ]; then
        return 0
    fi

    # Map the driver's max CUDA version to a PyTorch wheel index. The cuXYZ index
    # ships wheels built for that CUDA runtime; a driver is forward-compatible with
    # equal-or-older CUDA runtimes only.
    local cuda_major cuda_minor torch_index
    cuda_major=$(echo "${DRIVER_CUDA_VERSION:-}" | cut -d. -f1)
    cuda_minor=$(echo "${DRIVER_CUDA_VERSION:-0.0}" | cut -d. -f2)
    case "$cuda_major" in
        13|14|15) torch_index="cu130" ;;
        12)
            # cu124 runs on any CUDA 12.x driver; prefer cu128 on 12.8+ for newer GPUs.
            if [ "${cuda_minor:-0}" -ge 8 ] 2>/dev/null; then
                torch_index="cu128"
            else
                torch_index="cu124"
            fi
            ;;
        11)       torch_index="cu118" ;;
        *)        torch_index="" ;;
    esac

    if [ -z "$torch_index" ]; then
        print_info "Could not map driver CUDA '${DRIVER_CUDA_VERSION:-unknown}' to a torch wheel; using default PyPI torch."
        return 0
    fi

    print_step "Installing PyTorch for CUDA ${DRIVER_CUDA_VERSION} (index: ${torch_index})..."
    if $PYTHON_CMD -m pip install "torch>=2.0.0,<3.0" \
            --index-url "https://download.pytorch.org/whl/${torch_index}" > /dev/null 2>&1; then
        # Confirm torch can actually see the GPUs; if not, report the mismatch loudly.
        if $PYTHON_CMD -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" > /dev/null 2>&1; then
            print_success "PyTorch installed with working CUDA (${torch_index})"
        else
            print_warning "PyTorch installed (${torch_index}) but torch.cuda.is_available() is False — see docs/installation.md (CUDA matrix)."
        fi
    else
        print_warning "Could not install the ${torch_index} PyTorch wheel; falling back to default. See docs/installation.md."
    fi
}
install_compatible_torch

if [ "$MINIMAL_INSTALL" = true ]; then
    print_step "Installing minimal dependencies..."
    $PYTHON_CMD -m pip install -e . > /dev/null 2>&1
    $PYTHON_CMD -m pip install "bitsandbytes>=0.46.1" > /dev/null 2>&1
    print_success "Minimal installation complete"
else
    # Install core requirements
    if [ -f "requirements.txt" ]; then
        print_step "Installing core requirements..."
        echo ""
        # Show live output with green lines for each package
        $PYTHON_CMD -m pip install -r requirements.txt 2>&1 | while IFS= read -r line; do
            if [[ "$line" =~ ^Collecting || "$line" =~ ^Downloading || "$line" =~ ^Installing || "$line" =~ ^Successfully ]]; then
                echo -e "${GREEN}──${NC} ${DIM}${line}${NC}"
            elif [[ "$line" =~ ^Requirement ]]; then
                echo -e "${CYAN}──${NC} ${DIM}${line}${NC}"
            elif [[ "$line" =~ ERROR || "$line" =~ error ]]; then
                echo -e "${RED}──${NC} ${line}"
            else
                echo -e "${DIM}   ${line}${NC}"
            fi
        done
        print_success "Core requirements installed"
    else
        print_warning "requirements.txt not found"
    fi

    # Install development dependencies
    if [ "$INSTALL_DEV" = true ]; then
        if [ -f "requirements-dev.txt" ]; then
            print_step "Installing development dependencies..."
            $PYTHON_CMD -m pip install -r requirements-dev.txt > /dev/null 2>&1
            print_success "Development dependencies installed"
        else
            print_warning "requirements-dev.txt not found"
        fi
    fi

    # Install vLLM
    if [ "$INSTALL_VLLM" = true ]; then
        print_step "Installing vLLM..."
        print_info "vLLM provides 5-10x faster inference for production use"
        echo ""
        # Show live output with green lines for each package
        $PYTHON_CMD -m pip install vllm 2>&1 | while IFS= read -r line; do
            if [[ "$line" =~ ^Collecting || "$line" =~ ^Downloading || "$line" =~ ^Installing || "$line" =~ ^Successfully ]]; then
                echo -e "${GREEN}──${NC} ${DIM}${line}${NC}"
            elif [[ "$line" =~ ^Requirement ]]; then
                echo -e "${CYAN}──${NC} ${DIM}${line}${NC}"
            elif [[ "$line" =~ ERROR || "$line" =~ error ]]; then
                echo -e "${RED}──${NC} ${line}"
            else
                echo -e "${DIM}   ${line}${NC}"
            fi
        done
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            print_success "vLLM installed successfully"
            # vLLM pins one exact torch; a plain `pip install vllm` can replace the
            # driver-compatible torch we installed above with a newer-CUDA wheel.
            # Re-check and warn loudly rather than silently losing the GPU.
            if [ "$HAS_GPU" = true ] && ! $PYTHON_CMD -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" > /dev/null 2>&1; then
                print_warning "vLLM upgraded torch to a build that can't use this driver (torch.cuda.is_available() is False)."
                print_warning "Pick a vLLM release whose pinned torch matches CUDA ${DRIVER_CUDA_VERSION:-your driver} — see docs/installation.md (Installing vLLM)."
            fi
            echo ""
        else
            print_warning "vLLM installation failed - you can install it manually later"
        fi
    fi

    # Install the package itself
    print_step "Installing effGen package (pip install -e .)..."
    $PYTHON_CMD -m pip install -e . > /dev/null 2>&1
    print_success "effGen package installed in development mode"

    # Install bitsandbytes for 4-bit quantization support
    print_step "Installing bitsandbytes for 4-bit quantization..."
    $PYTHON_CMD -m pip install "bitsandbytes>=0.46.1" > /dev/null 2>&1
    print_success "bitsandbytes installed"
fi

echo ""
phase_delay

################################################################################
# Configuration Setup
################################################################################

print_header "⚙️  Configuration Setup"

# Create config directory
CONFIG_DIR="$PROJECT_ROOT/config"
mkdir -p "$CONFIG_DIR"

# Create default config if it doesn't exist
if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    print_step "Creating default configuration file..."
    cat > "$CONFIG_DIR/config.yaml" << 'EOF'
# effGen Configuration File

# Model settings
model:
  default_provider: "transformers"
  default_model: "Qwen/Qwen2.5-3B-Instruct"
  cache_dir: "./models"
  device: "auto"
  quantization: "4bit"

# Agent settings
agent:
  max_iterations: 10
  temperature: 0.7
  enable_sub_agents: true
  enable_memory: true

# Logging settings
logging:
  level: "INFO"
  log_dir: "./logs"
  console_output: true
  file_output: true
  structured: false

# Tool settings
tools:
  enable_code_execution: true
  enable_web_search: true
  sandbox_timeout: 30
  allowed_imports: ["math", "random", "json", "datetime"]

# Memory settings
memory:
  backend: "sqlite"
  db_path: "./memory.db"
  max_history: 100
  enable_long_term: true

# Performance settings
performance:
  batch_size: 1
  max_workers: 4
  enable_caching: true
  cache_dir: "./.cache"
EOF
    print_success "Default configuration created at config/config.yaml"
else
    print_info "Configuration file already exists"
fi

# Create .env file if it doesn't exist
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    print_step "Creating .env file for API keys..."
    cat > "$PROJECT_ROOT/.env" << 'EOF'
# API Keys for External Services
# Add your keys below (remove the # to uncomment)

# OpenAI API
#OPENAI_API_KEY=your-openai-key-here

# Anthropic API
#ANTHROPIC_API_KEY=your-anthropic-key-here

# Google Gemini API
#GOOGLE_API_KEY=your-google-key-here

# HuggingFace Token (for gated models)
#HF_TOKEN=your-huggingface-token-here

# Other Services
#SERPAPI_KEY=your-serpapi-key-here
#WANDB_API_KEY=your-wandb-key-here
EOF
    print_success ".env file created"
    print_info "  Edit .env to add your API keys"
else
    print_info ".env file already exists"
fi

# Note: logs, cache, and models directories will be created automatically when needed
print_success "Configuration complete"

echo ""
phase_delay

################################################################################
# Model Download (Optional)
################################################################################

if [ "$DOWNLOAD_MODELS" = true ]; then
    print_header "🤖 Model Download"

    if [ -f "$SCRIPT_DIR/download_models.py" ]; then
        print_step "Launching model downloader..."
        print_info "This will help you download recommended models"
        echo ""
        $PYTHON_CMD "$SCRIPT_DIR/download_models.py" --interactive
        echo ""
        print_success "Model download complete"
    else
        print_warning "download_models.py not found, skipping"
    fi

    echo ""
    phase_delay
fi

################################################################################
# Verification Tests
################################################################################

if [ "$SKIP_VERIFICATION" = false ]; then
    print_header "✅ Verification Tests"

    print_step "Testing core imports..."

    $PYTHON_CMD -c "
import sys

# Test effgen import
try:
    import effgen
    print('${GREEN}✓${NC} effgen module imported successfully')
    print('${BLUE}  ℹ${NC} Version: ' + str(getattr(effgen, '__version__', 'unknown')))
except ImportError as e:
    print('${RED}✗${NC} Failed to import effgen: ' + str(e))
    sys.exit(1)

# Test PyTorch
try:
    import torch
    print('${GREEN}✓${NC} PyTorch imported successfully')
    print('${BLUE}  ℹ${NC} Version: ' + torch.__version__)
    if torch.cuda.is_available():
        print('${GREEN}  ●${NC} CUDA available: ' + torch.cuda.get_device_name(0))
        print('${BLUE}  ℹ${NC} CUDA version: ' + torch.version.cuda)
    else:
        # Explain *why* — a torch-CUDA / driver mismatch is the common, fixable cause.
        try:
            from effgen.gpu.cuda_compat import get_cuda_status
            st = get_cuda_status()
            if st.mismatch and st.message:
                print('${YELLOW}  ⚠${NC} ' + st.message)
            else:
                print('${YELLOW}  ⚠${NC} CUDA not available (CPU mode)')
        except Exception:
            print('${YELLOW}  ⚠${NC} CUDA not available (CPU mode)')
except ImportError as e:
    print('${YELLOW}⚠${NC} PyTorch not installed: ' + str(e))

# Test Transformers
try:
    import transformers
    print('${GREEN}✓${NC} Transformers imported successfully')
    print('${BLUE}  ℹ${NC} Version: ' + transformers.__version__)
except ImportError as e:
    print('${YELLOW}⚠${NC} Transformers not installed: ' + str(e))

# Test vLLM (optional)
try:
    import vllm
    print('${GREEN}✓${NC} vLLM imported successfully')
    print('${BLUE}  ℹ${NC} Version: ' + vllm.__version__)
except ImportError:
    pass  # vLLM is optional

print('')
print('${GREEN}${BOLD}✓ All core components verified!${NC}')
"

    echo ""

    # Test simple agent creation
    print_step "Testing agent creation..."

    TEST_RESULT=$($PYTHON_CMD -c "
try:
    from effgen import Agent, load_model
    from effgen.core.agent import AgentConfig
    print('${GREEN}✓${NC} Agent creation test passed')
    print('0')
except Exception as e:
    print('${RED}✗${NC} Agent creation test failed: ' + str(e))
    print('1')
" | tail -1)

    if [ "$TEST_RESULT" = "0" ]; then
        print_success "Basic functionality verified"
    else
        print_warning "Some functionality tests failed (this may be normal)"
    fi

    echo ""
    phase_delay
fi

################################################################################
# Alpha Tests Information
################################################################################

print_header "📚 Examples Available"

print_info "Ready-to-run examples in examples/"
echo ""
print_feature "basic_agent.py - Simple agent example"
print_feature "code_assistant.py - Code generation agent"
print_feature "research_agent.py - Web research agent"
print_feature "multi_agent.py - Multi-agent orchestration"
print_feature "custom_tools.py - Creating custom tools"
echo ""
print_info "Run an example: python examples/basic_agent.py"
echo ""

phase_delay

################################################################################
# Installation Complete
################################################################################

if [ "$QUICK_MODE" = false ]; then
    clear_screen
fi
section_delay
print_success_big "INSTALLATION COMPLETE"
phase_delay

# Success message
echo -e "${GREEN}${BOLD}✨ effGen Framework Successfully Installed! ✨${NC}"
echo ""
section_delay

# Environment activation instructions
if [ "$USE_CONDA" = true ]; then
    echo -e "${WHITE}${BOLD}To use effGen:${NC}"
    echo ""
    line_delay
    echo "  1. Activate the conda environment:"
    line_delay
    echo -e "     ${CYAN}conda activate ${CONDA_ENV_NAME}${NC}"
    echo ""
    section_delay
fi

# Next steps
echo -e "${WHITE}${BOLD}Next Steps:${NC}"
echo ""
line_delay
echo "  1. Add your API keys to .env:"
line_delay
echo -e "     ${CYAN}nano .env${NC}"
echo ""
line_delay
echo "  2. Customize configuration:"
line_delay
echo -e "     ${CYAN}nano config/config.yaml${NC}"
echo ""
line_delay
echo "  3. Try a quick start example:"
line_delay
echo -e "     ${CYAN}python examples/basic_agent.py${NC}"
echo ""
line_delay
echo "  4. Use the CLI:"
line_delay
echo -e "     ${CYAN}effgen run \"What is 2+2?\" --model phi3-mini${NC}"
echo ""
line_delay
echo "  5. Read the documentation:"
line_delay
echo -e "     ${CYAN}https://effgen.org/docs/${NC}"
echo ""
section_delay

# Summary
echo -e "${WHITE}${BOLD}Installation Summary:${NC}"
echo ""
line_delay
if [ "$USE_CONDA" = true ]; then
    print_feature "Conda environment: ${CONDA_ENV_NAME}"
    line_delay
fi
print_feature "Framework: Installed in development mode"
line_delay
print_feature "Configuration: config/config.yaml"
line_delay
print_feature "API Keys: .env (please configure)"
line_delay
print_feature "Logs: logs/"
line_delay
print_feature "Models: models/"
line_delay
if [ "$INSTALL_VLLM" = true ]; then
    print_feature "vLLM: Installed for fast inference"
    line_delay
fi
echo ""
section_delay

# Final message
echo -e "${CYAN}${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
line_delay
echo -e "${CYAN}${BOLD}║                                                                ║${NC}"
line_delay
echo -e "${CYAN}${BOLD}║            🚀 Happy Building with effGen! 🚀               ║${NC}"
line_delay
echo -e "${CYAN}${BOLD}║                                                                ║${NC}"
line_delay
echo -e "${CYAN}${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
section_delay

# Save activation script for convenience
if [ "$USE_CONDA" = true ]; then
    cat > "$PROJECT_ROOT/activate.sh" << EOF
#!/bin/bash
# Quick activation script for effGen conda environment
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}
echo "✓ effGen environment activated"
EOF
    chmod +x "$PROJECT_ROOT/activate.sh"
    print_info "Quick activation script created: ./activate.sh"
    echo ""
    line_delay
fi
