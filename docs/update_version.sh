#!/bin/bash
# Script to update Doxyfile version from pyproject.toml

# Get the directory where this script is located (should be docs/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of docs/)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYPROJECT_FILE="$PROJECT_ROOT/pyproject.toml"
DOXYFILE="$SCRIPT_DIR/Doxyfile"

# Extract version from pyproject.toml
if [ ! -f "$PYPROJECT_FILE" ]; then
    echo "Error: pyproject.toml not found at $PYPROJECT_FILE" >&2
    exit 1
fi

VERSION=$(grep -E '^version\s*=' "$PYPROJECT_FILE" | sed -E 's/.*version\s*=\s*["\x27]([^"\x27]+)["\x27].*/\1/')

if [ -z "$VERSION" ]; then
    echo "Error: Could not extract version from pyproject.toml" >&2
    exit 1
fi

# Update Doxyfile
if [ ! -f "$DOXYFILE" ]; then
    echo "Error: Doxyfile not found at $DOXYFILE" >&2
    exit 1
fi

sed -i "s/^PROJECT_NUMBER\s*=\s*.*/PROJECT_NUMBER         = $VERSION/" "$DOXYFILE"

echo "Updated Doxyfile PROJECT_NUMBER to $VERSION"