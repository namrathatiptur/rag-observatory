#!/bin/bash

echo "=== SSH Public Key ==="
cat ~/.ssh/id_ed25519_github.pub 2>/dev/null || cat ~/.ssh/id_ed25519.pub 2>/dev/null
echo ""
echo "Add this SSH key to: https://github.com/settings/keys"
echo ""
echo "=== GPG Public Key ==="
GPG_KEY_ID=$(gpg --list-secret-keys --keyid-format LONG 2>/dev/null | grep -A 1 "^sec" | tail -1 | awk '{print $2}' | cut -d'/' -f2)
if [ ! -z "$GPG_KEY_ID" ]; then
    gpg --armor --export $GPG_KEY_ID
    echo ""
    echo "GPG Key ID: $GPG_KEY_ID"
    echo "Add this GPG key to: https://github.com/settings/gpg/new"
else
    echo "GPG key not found. Run: gpg --gen-key"
fi
