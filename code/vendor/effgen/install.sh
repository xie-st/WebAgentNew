#!/bin/bash

################################################################################
# effGen Installer (repo-root front-end)
#
# This is the one documented entry point for setting up effGen from a clone:
#   ./install.sh
# It parses the user-facing options below and then delegates the actual work to
# the CANONICAL install engine, scripts/install_effgen.sh, which owns the
# driver-aware PyTorch selection (so a CUDA-13 default wheel is never installed
# on a CUDA-12 driver). Verification is handled by the engine (or scripts/
# verify.sh for --skip-install). There is exactly one installer engine; this
# front-end and scripts/setup_and_verify.sh are thin wrappers around it.
#
# Usage:
#   ./install.sh                    # Standard installation
#   ./install.sh --quick            # Fast install (no animations)
#   ./install.sh --full             # Full install with all features
#   ./install.sh --help             # Show all options
#
################################################################################

set -e

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
NC='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'

################################################################################
# Configuration & Defaults
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Installation options
SKIP_INSTALL=false
SKIP_VERIFY=false
INSTALL_VLLM=false
DOWNLOAD_MODELS=false
INSTALL_DEV=false
VERBOSE=false
QUICK_MODE=false
FULL_MODE=false
SKIP_CONDA=false
MINIMAL_INSTALL=false
YES_TO_ALL=false

################################################################################
# UI Helper Functions
################################################################################

print_banner() {
    # Wiping the screen only makes sense on a terminal that can be wiped.
    # Without a TERM to describe it, `clear` fails with "TERM environment
    # variable not set." and, under `set -e`, ends the install on its very
    # first line — which is what a CI runner and a piped install both get.
    if [ "$QUICK_MODE" = false ] && [ -t 1 ] && [ -n "${TERM:-}" ] && [ "${TERM}" != "dumb" ]; then
        clear || true
    fi
    echo ""
    echo -e "${CYAN}${BOLD}+================================================================+${NC}"
    echo -e "${CYAN}${BOLD}|${NC}                                                                ${CYAN}${BOLD}|${NC}"
    echo -e "${CYAN}${BOLD}|${NC}              ${MAGENTA}${BOLD}effGen Framework Installer${NC}              ${CYAN}${BOLD}|${NC}"
    echo -e "${CYAN}${BOLD}|${NC}                                                                ${CYAN}${BOLD}|${NC}"
    echo -e "${CYAN}${BOLD}|${NC}        ${WHITE}Efficient Agents for Small Language Models${NC}        ${CYAN}${BOLD}|${NC}"
    echo -e "${CYAN}${BOLD}|${NC}                                                                ${CYAN}${BOLD}|${NC}"
    echo -e "${CYAN}${BOLD}+================================================================+${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}${BOLD}[OK]${NC} $1"
}

print_error() {
    echo -e "${RED}${BOLD}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}${BOLD}[WARN]${NC} $1"
}

print_info() {
    echo -e "${BLUE}${BOLD}[INFO]${NC} $1"
}

print_step() {
    echo -e "${WHITE}${BOLD}>>>${NC} $1"
}

print_section() {
    echo ""
    echo -e "${CYAN}${BOLD}--- $1 ---${NC}"
    echo ""
}

################################################################################
# Help Message
################################################################################

show_help() {
    cat << 'EOF'
effGen Master Installer

USAGE:
    ./install.sh [OPTIONS]

INSTALLATION MODES:
    ./install.sh                Standard install (core dependencies + verification)
    ./install.sh --quick        Same as standard, but no animations (CI-friendly)
    ./install.sh --full         Everything: vLLM, dev tools, model downloads

    Note: --quick and --full only differ in WHAT gets installed:
      - Standard/Quick: Core framework only
      - Full: Core + vLLM + dev tools + model downloads
    The --quick flag just disables animations for faster/cleaner output.

OPTIONS:
    Speed & Output:
        --quick             No animations, auto-confirm prompts (for CI/scripts)
        -y, --yes           Auto-confirm all prompts

    What to Install:
        --full              Install everything (vLLM + dev + models)
        --minimal           Minimal dependencies only
        --install-vllm      Add vLLM for faster inference
        --download-models   Download recommended models after install
        --dev               Add development dependencies (pytest, etc.)

    Skip Steps:
        --skip-install      Skip installation, only run verification
        --skip-verify       Skip verification after installation
        --skip-conda        Use system Python instead of conda

    Output:
        --verbose           Show detailed output
        --help, -h          Show this help message

EXAMPLES:
    # Standard install (recommended for most users)
    ./install.sh

    # CI/automation (same install, no animations)
    ./install.sh --quick

    # Full install with all optional features
    ./install.sh --full

    # Full install, no animations
    ./install.sh --full --quick

    # Just core framework, no conda
    ./install.sh --minimal --skip-conda

    # Just verify existing installation
    ./install.sh --skip-install

AFTER INSTALLATION:
    source activate.sh              # Activate environment
    python examples/basic_agent.py  # Run example
    effgen run "What is 2+2?"       # Use CLI

For more info: https://effgen.org/docs/
EOF
    exit 0
}

################################################################################
# Argument Parsing
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            YES_TO_ALL=true
            shift
            ;;
        --full)
            FULL_MODE=true
            INSTALL_VLLM=true
            INSTALL_DEV=true
            DOWNLOAD_MODELS=true
            shift
            ;;
        --minimal)
            MINIMAL_INSTALL=true
            shift
            ;;
        -y|--yes)
            YES_TO_ALL=true
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --install-vllm)
            INSTALL_VLLM=true
            shift
            ;;
        --download-models)
            DOWNLOAD_MODELS=true
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --skip-conda)
            SKIP_CONDA=true
            shift
            ;;
        --verbose)
            VERBOSE=true
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
# Pre-flight Checks
################################################################################

check_prerequisites() {
    print_section "Checking Prerequisites"

    local has_errors=false

    # Check for conda (optional but recommended)
    local has_conda=false
    if [ "$SKIP_CONDA" = false ]; then
        if command -v conda &> /dev/null; then
            print_success "Conda found: $(conda --version)"
            has_conda=true
        else
            print_warning "Conda not found - will use system Python"
            SKIP_CONDA=true
        fi
    else
        print_info "Skipping conda (using system Python)"
    fi

    # Check for Python
    if command -v python3 &> /dev/null; then
        local py_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        local py_major=$(echo $py_version | cut -d. -f1)
        local py_minor=$(echo $py_version | cut -d. -f2)

        if [ "$py_major" -ge 3 ] && [ "$py_minor" -ge 11 ]; then
            print_success "Python $py_version found"
        else
            if [ "$has_conda" = true ]; then
                print_warning "System Python is $py_version — conda will create a 3.11 environment"
            else
                print_error "Python 3.11+ required (found $py_version)"
                has_errors=true
            fi
        fi
    else
        if [ "$has_conda" = true ]; then
            print_warning "System Python not found — conda will provide Python 3.11"
        else
            print_error "Python 3 not found"
            has_errors=true
        fi
    fi

    # Check for git
    if command -v git &> /dev/null; then
        print_success "Git found"
    else
        print_warning "Git not found (optional)"
    fi

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        print_success "NVIDIA GPU detected ($gpu_count GPU(s))"
    else
        print_info "No NVIDIA GPU detected - will use CPU mode"
    fi

    if [ "$has_errors" = true ]; then
        print_error "Prerequisites check failed"
        exit 1
    fi

    echo ""
}

################################################################################
# Installation Summary
################################################################################

show_install_summary() {
    print_section "Installation Summary"

    echo -e "${WHITE}The following will be installed:${NC}"
    echo ""

    if [ "$SKIP_CONDA" = false ]; then
        echo -e "  ${GREEN}[*]${NC} Conda environment: effgen"
    else
        echo -e "  ${DIM}[ ]${NC} ${DIM}Conda environment (skipped)${NC}"
    fi

    echo -e "  ${GREEN}[*]${NC} Core dependencies (PyTorch, Transformers)"
    echo -e "  ${GREEN}[*]${NC} effGen framework"
    echo -e "  ${GREEN}[*]${NC} Configuration files"

    if [ "$INSTALL_VLLM" = true ]; then
        echo -e "  ${GREEN}[*]${NC} vLLM (fast inference)"
    else
        echo -e "  ${DIM}[ ]${NC} ${DIM}vLLM (use --install-vllm to add)${NC}"
    fi

    if [ "$INSTALL_DEV" = true ]; then
        echo -e "  ${GREEN}[*]${NC} Development dependencies"
    else
        echo -e "  ${DIM}[ ]${NC} ${DIM}Dev dependencies (use --dev to add)${NC}"
    fi

    if [ "$DOWNLOAD_MODELS" = true ]; then
        echo -e "  ${GREEN}[*]${NC} Model downloads"
    else
        echo -e "  ${DIM}[ ]${NC} ${DIM}Models (use --download-models to add)${NC}"
    fi

    if [ "$SKIP_VERIFY" = false ]; then
        echo -e "  ${GREEN}[*]${NC} Installation verification"
    fi

    echo ""

    # Confirm unless --yes was specified. Only ask when there is somebody to
    # answer: run from CI, from another script, or with stdin redirected, this
    # read returns non-zero at once and ends the install at the confirmation
    # step. Nobody who invoked the installer needs it confirmed back to them,
    # so with no terminal it goes ahead — the same thing --yes asks for.
    if [ "$YES_TO_ALL" = false ] && [ "$QUICK_MODE" = false ] && [ -t 0 ]; then
        read -p "$(echo -e ${YELLOW}Proceed with installation? [Y/n]:${NC} )" -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]?$ ]]; then
            print_info "Installation cancelled"
            exit 0
        fi
    fi
}

################################################################################
# Export Settings for Sub-scripts
################################################################################

export_settings() {
    # Export settings for child scripts to inherit
    export EFFGEN_QUICK_MODE="$QUICK_MODE"
    export EFFGEN_YES_TO_ALL="$YES_TO_ALL"
}

################################################################################
# Run Installation
################################################################################

run_installation() {
    print_section "Running Installation"

    # The canonical install engine (owns driver-aware torch selection).
    local CMD="$SCRIPT_DIR/scripts/install_effgen.sh"
    if [ ! -f "$CMD" ]; then
        print_error "Canonical installer not found: $CMD"
        print_info "Make sure you're running from the effGen repository root"
        exit 1
    fi

    # --skip-install: run verification only, never the install engine.
    if [ "$SKIP_INSTALL" = true ]; then
        local VCMD="$SCRIPT_DIR/scripts/verify.sh"
        if [ -f "$VCMD" ]; then
            local VARGS=""
            [ "$QUICK_MODE" = true ] && VARGS="$VARGS --quick"
            [ "$VERBOSE" = true ] && VARGS="$VARGS --verbose"
            print_info "Skipping install; running verification only: bash $VCMD$VARGS"
            echo ""
            if ! bash "$VCMD" $VARGS; then
                print_error "Verification failed"
                exit 1
            fi
        else
            print_warning "Verification script not found: $VCMD"
        fi
        return
    fi

    # Map this front-end's options onto the canonical engine's flags.
    local ARGS=""
    [ "$QUICK_MODE" = true ]      && ARGS="$ARGS --quick"
    [ "$SKIP_CONDA" = true ]      && ARGS="$ARGS --skip-conda"
    [ "$MINIMAL_INSTALL" = true ] && ARGS="$ARGS --minimal"
    [ "$INSTALL_VLLM" = true ]    && ARGS="$ARGS --install-vllm"
    [ "$DOWNLOAD_MODELS" = true ] && ARGS="$ARGS --download-models"
    [ "$INSTALL_DEV" = true ]     && ARGS="$ARGS --dev"
    [ "$SKIP_VERIFY" = true ]     && ARGS="$ARGS --skip-verification"

    print_info "Running canonical installer: bash $CMD$ARGS"
    echo ""

    if ! bash "$CMD" $ARGS; then
        print_error "Installation failed"
        exit 1
    fi
}

################################################################################
# Final Instructions
################################################################################

show_final_instructions() {
    if [ "$QUICK_MODE" = true ]; then
        echo ""
        print_success "Installation complete!"
        echo ""
        echo "Quick start:"
        echo "  source activate.sh && python examples/basic_agent.py"
        echo ""
    fi
    # The install engine shows detailed instructions, so we only add brief
    # instructions for quick mode
}

################################################################################
# Main Execution
################################################################################

main() {
    # Show banner
    print_banner

    # Quick mode notice
    if [ "$QUICK_MODE" = true ]; then
        print_info "Running in quick mode (minimal output)"
        echo ""
    fi

    # Full mode notice
    if [ "$FULL_MODE" = true ]; then
        print_info "Running full installation (includes vLLM, dev tools, models)"
        echo ""
    fi

    # Run pre-flight checks
    check_prerequisites

    # Show what will be installed
    if [ "$SKIP_INSTALL" = false ]; then
        show_install_summary
    fi

    # Export settings for sub-scripts
    export_settings

    # Run installation
    run_installation

    # Show final instructions
    show_final_instructions
}

################################################################################
# Execute
################################################################################

main
