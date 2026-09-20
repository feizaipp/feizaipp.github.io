---
layout:     post
title:      Install and Speed Up Homebrew in Mainland China
subtitle:   A practical macOS guide to installation, TUNA mirrors, verification, and recovery
date:       2026-09-20
author:     feizaipp
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - Homebrew
    - macOS
    - China
    - Developer Tools
---

Homebrew is the most convenient package manager for macOS, but a default installation and subsequent package downloads can be slow or unreliable from mainland China. The usual cause is access to GitHub, Homebrew's formula API, or the GitHub Container Registry used for prebuilt packages ("bottles").

This guide uses the [Tsinghua University TUNA mirror](https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/), a widely used academic mirror. It covers Apple Silicon and Intel Macs, assumes the default `zsh` shell, and keeps the configuration easy to undo.

> A mirror is part of the software supply chain. Only use a mirror you trust. Homebrew explicitly warns that a custom Git remote is granted the same level of trust as Homebrew itself. Prefer HTTPS and verify that the mirror is maintained by its stated operator.

## 1. Prerequisites

Install Apple's Command Line Tools if they are not already present:

```bash
xcode-select --install
```

Confirm your shell and CPU architecture:

```bash
echo "$SHELL"
uname -m
```

Most current macOS installations use `zsh`. Apple Silicon reports `arm64`; Intel Macs report `x86_64`.

## 2. Install Homebrew with a mirror

Set the Git mirror variables in the current terminal first. Homebrew's installer supports these variables for installations where GitHub access is problematic.

```bash
export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/brew.git"
export HOMEBREW_CORE_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-core.git"
export HOMEBREW_API_DOMAIN="https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles/api"
export HOMEBREW_BOTTLE_DOMAIN="https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles"
```

Then fetch the installer from the same mirror and run it:

```bash
git clone --depth=1 https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/install.git brew-install
/bin/bash brew-install/install.sh
rm -rf brew-install
```

Homebrew installs into `/opt/homebrew` on Apple Silicon and `/usr/local` on Intel Macs. The installer prints the exact post-installation command for your machine. On an Apple Silicon Mac using `zsh`, it is normally:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Do not copy the Apple Silicon path to an Intel Mac. Use the command printed by the installer instead.

## 3. Make the download acceleration persistent

Homebrew 4 and later normally reads formula and cask metadata through its JSON API, so the API and bottle settings are the important part for ordinary `brew install` and `brew upgrade` use. Add the following lines once to `~/.zprofile`:

```bash
cat >> ~/.zprofile <<'EOF'

# Homebrew mirrors for mainland China (Tsinghua TUNA)
export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/brew.git"
export HOMEBREW_API_DOMAIN="https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles/api"
export HOMEBREW_BOTTLE_DOMAIN="https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles"
EOF

source ~/.zprofile
```

`HOMEBREW_API_DOMAIN` speeds up formula and cask metadata. `HOMEBREW_BOTTLE_DOMAIN` speeds up precompiled package downloads. The Git remote helps `brew update` when the Homebrew repository itself needs to be fetched.

If you use `brew cat`, contribute to Homebrew, or deliberately keep `homebrew/core` and `homebrew/cask` checked out as taps, also add this line:

```bash
export HOMEBREW_CORE_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-core.git"
```

For ordinary users, there is no reason to set `HOMEBREW_NO_INSTALL_FROM_API=1`: doing so reverts to the older, clone-based workflow and increases update traffic.

## 4. Configure an existing installation

If Homebrew is already installed, add the persistent settings from the previous section, update the main repository remote, and refresh metadata:

```bash
git -C "$(brew --repo)" remote set-url origin \
  https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/brew.git
brew update
```

If `homebrew/core` or `homebrew/cask` appears in `brew tap`, its remote can be changed explicitly:

```bash
brew tap --custom-remote homebrew/core \
  https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-core.git
brew tap --custom-remote homebrew/cask \
  https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-cask.git
```

Those last two commands are optional. Modern Homebrew installations often do not have either tap cloned locally.

## 5. Verify the configuration

Run an update and install a small package:

```bash
brew update
brew install wget
brew config
```

During the download, URLs should point at `mirrors.tuna.tsinghua.edu.cn` for Homebrew metadata and bottles. `brew config` is useful for confirming the active prefix, architecture, and Homebrew version.

## 6. What a mirror cannot accelerate

Some casks download their application archives from a vendor site or GitHub Releases rather than from Homebrew bottles. A Homebrew mirror cannot change those external URLs. If these downloads remain slow, configure a reliable, trusted network proxy for your terminal or download the application from its official website.

Avoid random “one-line acceleration scripts” that replace formulae, rewrite downloads, or silently add third-party taps. They can make updates faster, but they expand the trust boundary far beyond a normal Homebrew mirror.

## 7. Return to the official Homebrew sources

Remove the `HOMEBREW_*` mirror exports from `~/.zprofile`, then run:

```bash
unset HOMEBREW_BREW_GIT_REMOTE
unset HOMEBREW_CORE_GIT_REMOTE
unset HOMEBREW_API_DOMAIN
unset HOMEBREW_BOTTLE_DOMAIN

git -C "$(brew --repo)" remote set-url origin \
  https://github.com/Homebrew/brew.git

brew tap --custom-remote homebrew/core \
  https://github.com/Homebrew/homebrew-core.git 2>/dev/null || true
brew tap --custom-remote homebrew/cask \
  https://github.com/Homebrew/homebrew-cask.git 2>/dev/null || true

brew update
```

Start a new terminal afterwards, or run `source ~/.zprofile`, so the removed variables do not remain in the current shell.

## References

* [Homebrew installation documentation](https://docs.brew.sh/Installation)
* [Homebrew environment variables](https://docs.brew.sh/Manpage)
* [TUNA Homebrew mirror guide](https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/)
* [TUNA Homebrew bottles guide](https://mirrors.tuna.tsinghua.edu.cn/help/homebrew-bottles/)
