# GitHub Access & Dual-Repository Architecture Protocol

> **CRITICAL DIRECTIVE FOR ALL AI AGENTS & CONTRIBUTORS:**  
> This project operates under a strict **Dual-Repository Topology**. Every modification, feature addition, bug fix, and architectural document MUST be synchronized across both remotes according to the rules defined below.

---

## 1. Repository Topology & Roles

| Remote Name | Repository URL | Access Type | Role & Responsibility |
|---|---|---|---|
| **`origin`** | `https://github.com/JackPro2121/podcasts-clips-to-social.git` | **Public Production** | • Hosts the automated GitHub Actions CI/CD pipeline (`podcast_clipper.yml`, `streamer_clipper.yml`).<br>• **MUST REMAIN 100% CLEAN** of all credentials, API tokens, and secrets.<br>• Runs autonomous scheduling, video transcoding, and live social publishing. |
| **`private`** | `https://github.com/ranajawadai/Private-Podcast-Clips-to-Social.git` | **Private Backup** | • Full, private mirror and long-term disaster recovery repository.<br>• Owned by `ranajawadai`.<br>• Preserves all architectural specs, engineering plans, and code history. |

---

## 2. Mandatory Dual-Push Protocol

Whenever you make any commit or code change, you **MUST** push to **BOTH** remotes:

```bash
# Step 1: Push to Primary Production Remote
git push origin main

# Step 2: Push to Private Backup Remote
git push private main
```

---

## 3. Strict Security & Zero-Credential Governance

### Production Cleanliness Invariant (`JackPro2121`)
- **NEVER** commit hardcoded API keys, tokens, cookies, or secrets to the repository.
- The following credentials belong **EXCLUSIVELY** in GitHub Secrets (`Settings -> Secrets and variables -> Actions`):
  - `GEMINI_API_KEY`
  - `BUFFER_ACCESS_TOKEN` / `BUFFER_CHANNEL_IDS` (V1 Podcast only)
  - `STREAMER_BUFFER_ACCESS_TOKEN` / `STREAMER_BUFFER_CHANNEL_IDS` (V2 Streamer only)
  - `APIFY_API_TOKEN`
  - `FIRECRAWL_API_KEY`
  - `GITHUB_TOKEN`
- Local testing credentials must reside in `.env`, which is strictly ignored by `.gitignore`.

---

## 4. GitHub CLI (`gh`) Account Management

Both accounts are authenticated on the local development environment:
- Active account switching:
  ```bash
  # Switch to production account
  gh auth switch --user JackPro2121

  # Switch to backup account
  gh auth switch --user ranajawadai
  ```
- Check authentication status:
  ```bash
  gh auth status
  ```

---

## 5. Summary of Remotes

```bash
$ git remote -v
origin   https://github.com/JackPro2121/podcasts-clips-to-social.git (fetch)
origin   https://github.com/JackPro2121/podcasts-clips-to-social.git (push)
private  https://github.com/ranajawadai/Private-Podcast-Clips-to-Social.git (fetch)
private  https://github.com/ranajawadai/Private-Podcast-Clips-to-Social.git (push)
```
