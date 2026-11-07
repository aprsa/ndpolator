#!/bin/bash
# build_docs.sh - Build documentation with doxygen.
#
# The project version is defined once, in pyproject.toml. The Doxyfile reads
# it from the NDPOLATOR_VERSION environment variable, which is exported here.

set -e  # Exit on any error

# Get the script directory (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

NDPOLATOR_VERSION="$(grep -E '^version\s*=' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/')"
if [ -z "$NDPOLATOR_VERSION" ]; then
    echo "Error: could not extract version from pyproject.toml" >&2
    exit 1
fi
export NDPOLATOR_VERSION

echo "Building ndpolator $NDPOLATOR_VERSION documentation..."
doxygen docs/Doxyfile

echo "Documentation built successfully!"
echo "Open docs/html/index.html to view the documentation."
