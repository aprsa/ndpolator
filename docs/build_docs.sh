#!/bin/bash
# build_docs.sh - Build documentation with automatically updated version

set -e  # Exit on any error

cd "$(dirname "$0")"

echo "Building ndpolator documentation..."

# Update version from pyproject.toml
echo "Updating version..."
./update_version.sh

# Build documentation
echo "Running doxygen..."
doxygen Doxyfile

echo "Documentation built successfully!"
echo "Open docs/html/index.html to view the documentation."