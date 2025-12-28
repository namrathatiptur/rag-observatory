#!/bin/bash
# Note: GPG keys cannot be changed after creation
# You would need to create a new GPG key with the correct email
# Or add an additional user ID to the existing key

echo "Current GPG key email:"
gpg --list-secret-keys --keyid-format LONG | grep -A 1 "^uid"

echo ""
echo "To add your email to the GPG key, run:"
echo "gpg --edit-key $(gpg --list-secret-keys --keyid-format LONG | grep '^sec' | head -1 | awk '{print $2}' | cut -d'/' -f2)"
echo "Then type: adduid"
echo "Enter: namrathatm2018@gmail.com"
echo "Then: save"
