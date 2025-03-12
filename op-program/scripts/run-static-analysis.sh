#!/usr/bin/env bash
set -x
set -e  # Exit immediately if a command exits with a non-zero status

# Usage function
usage() {
    echo "Usage: $0 <vm-profile> <baseline-report>"
    echo "Valid profiles: cannon-singlethreaded-32, cannon-multithreaded-64"
    exit 1
}

# Validate input
if [[ $# -ne 2 ]]; then
    usage
fi

VM_PROFILE=$1
BASELINE_REPORT=$2

if [[ "$VM_PROFILE" != "cannon-singlethreaded-32" && "$VM_PROFILE" != "cannon-multithreaded-64" ]]; then
    echo "Error: Invalid vm-profile '$VM_PROFILE'"
    usage
fi

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
BIN_DIR="bin"
ANALYZER_BIN="${BIN_DIR}/analyzer"

# Normalize architecture naming
case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
esac

if ! command -v llvm-objdump &>/dev/null; then
    echo "❌ Error: llvm-objdump is required but not found."
    echo "Please install it using one of the following commands, based on your OS:"
    echo "  Ubuntu/Debian: sudo apt-get install -y llvm"
    echo "  Fedora: sudo dnf install -y llvm"
    echo "  Arch Linux: sudo pacman -Sy llvm"
    echo "  macOS (Homebrew): brew install llvm"
    exit 1
fi

echo "✅ llvm-objdump found at $(which llvm-objdump)"

# Define the binary based on OS and ARCH
ANALYZER_VERSION="v1.1.0"
ANALYZER_URL="https://github.com/ChainSafe/vm-compat/releases/download/${ANALYZER_VERSION}/analyzer-${OS}-${ARCH}"

# Fetch Analyzer if not present
if [[ -f "$ANALYZER_BIN" ]]; then
    echo "Analyzer binary already exists, skipping download."
else
    echo "Fetching Analyzer Binary for ${OS}-${ARCH}..."
    mkdir -p "$BIN_DIR"
    curl -L -o "$ANALYZER_BIN" "$ANALYZER_URL"
    chmod +x "$ANALYZER_BIN"
fi

# Run the analyzer
echo "Running analysis with VM profile: $VM_PROFILE using baseline report: $BASELINE_REPORT..."
OUTPUT_FILE=$(mktemp)

"$ANALYZER_BIN" analyze --with-trace=true --format=json --vm-profile "$VM_PROFILE" --baseline-report "$BASELINE_REPORT" --report-output-path "$OUTPUT_FILE" ./client/cmd/main.go

# Check if JSON output contains any issues
ISSUE_COUNT=$(jq 'length' "$OUTPUT_FILE")

if [ "$ISSUE_COUNT" -gt 0 ]; then
    echo "❌ Analysis found $ISSUE_COUNT issues!"
    cat "$OUTPUT_FILE"
    rm -f "$OUTPUT_FILE"
    exit 1
else
    echo "✅ No issues found."
    rm -f "$OUTPUT_FILE"
    exit 0
fi
