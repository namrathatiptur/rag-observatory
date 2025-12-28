# SSH and GPG Keys Setup

## SSH Key

Your SSH public key:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDEe6YZ3RGg74HQkQlqdT4sEsdGLz4nznO5vl95wzrJu namrathatiptur@users.noreply.github.com
```

**Add to GitHub:**
1. Go to: https://github.com/settings/keys
2. Click "New SSH key"
3. Title: "RAG Observatory Mac"
4. Paste the key above
5. Click "Add SSH key"

## GPG Key

Your GPG public key (run this command to view):
```bash
gpg --armor --export $(gpg --list-secret-keys --keyid-format LONG | grep "^sec" | head -1 | awk '{print $2}' | cut -d'/' -f2)
```

**Add to GitHub:**
1. Copy your GPG public key (command above)
2. Go to: https://github.com/settings/gpg/new
3. Paste the key
4. Click "Add GPG key"

## Test Setup

After adding keys to GitHub:

1. **Test SSH:**
   ```bash
   ssh -T git@github.com
   ```
   Should see: "Hi namrathatiptur! You've successfully authenticated..."

2. **Test GPG signing:**
   ```bash
   git commit --allow-empty -m "Test signed commit"
   git log --show-signature -1
   ```
   Should see "Good signature" in the output

## Current Configuration

- SSH key: ~/.ssh/id_ed25519_github
- SSH config: ~/.ssh/config (configured for github.com)
- Git remote: Updated to SSH (git@github.com:namrathatiptur/rag-observatory.git)
- GPG signing: Enabled in git config
