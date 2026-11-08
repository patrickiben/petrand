#!/usr/bin/env bash
# ============================================================================
# Launch_PetRand.command (macOS Double-Click Launcher)
# Double-click this file in Finder to launch the unified PetRand Console,
# Everyday Randomness Studio, and the Zoomies! game in your browser.
# ============================================================================

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

PORT=8080
echo "================================================================"
echo " Starting PetRand // Everyday Randomness Studio & Zoomies! "
echo "================================================================"
echo "  - Pet Arena & Everyday Studio : http://127.0.0.1:${PORT}/"
echo "  - Zoomies! game: http://127.0.0.1:${PORT}/zoomies"
echo "================================================================"

# Automatically open the browser after 1.2 seconds
(sleep 1.2 && open "http://127.0.0.1:${PORT}/") &

python3 -m petrand.cli serve --host 127.0.0.1 --port "${PORT}" --sensor sim
