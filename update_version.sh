#!/bin/bash

# Ground-truth project version is in pyproject.toml
# This script updates __init__.py and Doxyfile to match that version
# The script is run automatically by build_docs.sh and CI workflows

# Get the directory where this script is located (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYPROJECT_FILE="$PROJECT_ROOT/pyproject.toml"
DOXYFILE="$PROJECT_ROOT/docs/Doxyfile"

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

# Update Python __init__.py version
INIT_FILE="$PROJECT_ROOT/ndpolator/__init__.py"
if [ -f "$INIT_FILE" ]; then
    sed -i "s/^__version__\s*=\s*['\"][^'\"]*['\"]/__version__ = '$VERSION'/" "$INIT_FILE"
    echo "Updated Python __version__ to $VERSION"
else
    echo "Warning: $INIT_FILE not found" >&2
fi

echo "Updated Doxyfile PROJECT_NUMBER to $VERSION"