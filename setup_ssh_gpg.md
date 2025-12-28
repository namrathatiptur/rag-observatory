# SSH and GPG Setup Instructions

## SSH Key Setup

Your SSH public key has been generated. Add it to GitHub:

1. Copy your SSH public key:
   ```bash
   cat ~/.ssh/id_ed25519_github.pub
   ```

2. Go to GitHub: https://github.com/settings/keys
3. Click "New SSH key"
4. Paste the key and save

## GPG Key Setup

Your GPG key has been generated. Add it to GitHub:

1. Copy your GPG public key:
   ```bash
   gpg --armor --export $(gpg --list-secret-keys --keyid-format LONG | grep -A 1 "^sec" | tail -1 | awk '{print $2}' | cut -d'/' -f2)
   ```

2. Go to GitHub: https://github.com/settings/gpg/new
3. Paste the key and save

## Test SSH Connection

After adding the SSH key to GitHub, test the connection:
```bash
ssh -T git@github.com
```

You should see: "Hi namrathatiptur! You've successfully authenticated..."

## Verify GPG Signing

Test that commits are signed:
```bash
git commit --allow-empty -m "Test signed commit"
git log --show-signature -1
```

## Current Configuration

- SSH key: ~/.ssh/id_ed25519_github
- SSH config: ~/.ssh/config
- GPG signing: Enabled
- Remote URL: Updated to SSH
