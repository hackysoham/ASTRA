# GitHub Repository Setup — Team Astra

**NSSC 2026** | Mars HiRISE Unsupervised Anomaly Detection

---

## Step 1: Create a Private Repository

1. Go to [https://github.com/new](https://github.com/new)
2. Set the following:
   - **Repository name:** `astra-hirise-anomaly`
   - **Description:** `Mars HiRISE Unsupervised Anomaly Detection Pipeline — Team Astra, NSSC 2026, IIT Kharagpur`
   - **Visibility:** ⚠️ **Private** (required by competition rules)
   - **Initialize with:** Do NOT add README (we have our own)
3. Click **Create repository**

## Step 2: Push Local Code

```bash
cd e:\ASTRA

git init
git add .
git commit -m "Initial commit: Team Astra NSSC 2026 pipeline

- Phase 1: Convolutional/Variational Autoencoder (5 versions)
- Phase 2: Isolation Forest with metadata fusion & statistical thresholding
- Phase 3: Reconstruction error heatmaps & geological analysis
- Phase 4: Architecture design journal (5 iterations)

Lead: "

git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/astra-hirise-anomaly.git
git push -u origin main
```

## Step 3: Add Mandatory Collaborators

> ⚠️ **All collaborators below MUST be added** per competition requirements.

Go to **Settings → Collaborators** (or navigate to `https://github.com/<YOUR-USERNAME>/astra-hirise-anomaly/settings/access`), then add each of the following GitHub usernames:

| # | GitHub Username | Action |
|---|-----------------|--------|
| 1 | `SarthakXSingh09` | Click "Add people" → Enter username → Invite |
| 2 | `R15HV` | Click "Add people" → Enter username → Invite |
| 3 | `Shivam3473` | Click "Add people" → Enter username → Invite |
| 4 | `aditohates-bugs` | Click "Add people" → Enter username → Invite |
| 5 | `Dhairya646` | Click "Add people" → Enter username → Invite |

### Quick CLI Method (GitHub CLI)

If you have the GitHub CLI (`gh`) installed:

```bash
gh repo create astra-hirise-anomaly --private --source=. --push

# Add collaborators
gh api repos/<YOUR-USERNAME>/astra-hirise-anomaly/collaborators/SarthakXSingh09 -X PUT
gh api repos/<YOUR-USERNAME>/astra-hirise-anomaly/collaborators/R15HV -X PUT
gh api repos/<YOUR-USERNAME>/astra-hirise-anomaly/collaborators/Shivam3473 -X PUT
gh api repos/<YOUR-USERNAME>/astra-hirise-anomaly/collaborators/aditohates-bugs -X PUT
gh api repos/<YOUR-USERNAME>/astra-hirise-anomaly/collaborators/Dhairya646 -X PUT
```

## Step 4: Set Up `.gitignore`

Add a `.gitignore` to prevent large files from being tracked:

```gitignore
# Data (too large for git)
data/images/
*.npy

# Model checkpoints
outputs/models/*.pth

# Python
__pycache__/
*.pyc
.venv/
venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

## Step 5: Verify

- [ ] Repository is **private**
- [ ] All 5 collaborators have been invited
- [ ] Code is pushed to `main` branch
- [ ] `.gitignore` is configured
- [ ] README.md is visible on the repo page

---

**Team Astra** — 
