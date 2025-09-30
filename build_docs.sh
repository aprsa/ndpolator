#!/bin/bash
# build_docs.sh - Build documentation with automatically updated version

set -e  # Exit on any error

# Get the script directory (project root)  
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to project root (we're already there, but be explicit)
cd "$PROJECT_ROOT"

echo "Building ndpolator documentation..."

# Update version from pyproject.toml
echo "Updating version..."
bash update_version.sh

# Build documentation
echo "Running doxygen..."
doxygen docs/Doxyfile

echo "Documentation built successfully!"
echo "Open docs/html/index.html to view the documentation."