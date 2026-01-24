#!/bin/bash
# Install Triton 3.6.0 (latest) with SM_121 support
# This replaces the old Triton 3.5.1

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Installing Triton 3.6.0 (Latest) for SM_121 Support   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "Removing old Triton..."
pip uninstall -y triton || true

echo ""
echo "Installing Triton 3.6.0 from PyPI..."
pip install triton==3.6.0

echo ""
echo "Verifying installation..."
python3 -c "import triton; print(f'✅ Triton {triton.__version__} installed')"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Triton 3.6.0 Installation Complete            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Key improvements in 3.6.0:"
echo "  • TMA gather4 support for SM_120 and SM_121 (PR #8498)"
echo "  • Blackwell architecture support"
echo "  • dot_scaled improvements"
echo ""
