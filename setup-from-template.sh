#!/usr/bin/env bash
# Run this after cloning the template to replace all placeholders.
# Usage: ./setup-from-template.sh <package_name> "<description>"

set -e

NAME=$1
DESC=$2

if [ -z "$NAME" ] || [ -z "$DESC" ]; then
  echo "Usage: ./setup-from-template.sh <package_name> \"<description>\""
  exit 1
fi

# Rename package directory
mv PACKAGE_NAME "$NAME"

# Replace in all files
find . -type f -not -path './.git/*' -not -name 'setup-from-template.sh' | while read f; do
  if file "$f" | grep -q text; then
    sed -i '' "s/PACKAGE_NAME/$NAME/g" "$f"
    sed -i '' "s/PACKAGE_DESCRIPTION/$DESC/g" "$f"
  fi
done

# Set up the development environment
source scripts/enable-venv.sh

# Verify everything passes
make all

# Initial commit
git add -A
git commit -m "Initial scaffold from python-lib-template"

# Clean up this script
git rm setup-from-template.sh
git commit -m "Remove template setup script"

echo ""
echo "Done! '$NAME' is ready."
