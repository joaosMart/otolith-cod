#!/bin/bash
# create_issues.sh
# Run: chmod +x create_issues.sh && ./create_issues.sh

REPO="joaosmart/otolith-cod"  # Change this

gh issue create --repo $REPO \
  --title "Setup & Data Preparation" \
  --label "setup,data" \
  --body "## Tasks
- [ ] Create conda environment (torch, transformers>=4.49, scikit-learn, pandas, numpy, PIL, matplotlib)
- [ ] Organize cod dataset into standard structure
- [ ] Analyze age distribution, decide on clipping/merging strategy
- [ ] Implement stratified 10-fold split: 65% train / 15% val / 20% test
- [ ] Store splits as CSV for reproducibility

## Acceptance
Can load all images and labels; splits are deterministic with fixed seed"

gh issue create --repo $REPO \
  --title "Feature Extraction with SigLIP2" \
  --label "feature-extraction,core" \
  --body "## Tasks
- [ ] Implement \`extract_features.py\`
- [ ] Use \`google/siglip2-so400m-patch14-384\` (1152-dim output)
- [ ] Batch processing for efficiency
- [ ] Save embeddings as \`.npy\` files
- [ ] Also extract CLIP ViT-L/14@336px for baseline comparison

## Code Reference
\`\`\`python
from transformers import AutoModel, AutoProcessor
ckpt = \"google/siglip2-so400m-patch14-384\"
model = AutoModel.from_pretrained(ckpt).eval()
processor = AutoProcessor.from_pretrained(ckpt)
# embeddings shape: (N, 1152)
\`\`\`

## Acceptance
All images embedded; features saved and reloadable"

gh issue create --repo $REPO \
  --title "Ridge Regression Model" \
  --label "model,core" \
  --body "## Tasks
- [ ] Implement \`train_ridge.py\`
- [ ] Input: SigLIP2 features (1152-dim)
- [ ] Model: \`sklearn.linear_model.Ridge(alpha=6.0)\`
- [ ] Grid search α in [0.1, 20] with CV on training set
- [ ] Round predictions to nearest integer for classification
- [ ] Report: Accuracy, ±1 Accuracy, RMSE per fold

## Acceptance
10-fold CV results saved to CSV"

gh issue create --repo $REPO \
  --title "SVC Multiclass Model (Optional)" \
  --label "model,optional" \
  --body "## Tasks
- [ ] Implement \`train_svc.py\`
- [ ] Model: \`sklearn.svm.SVC(kernel='linear', C=0.1)\`
- [ ] Compare with Ridge regression

## Acceptance
Classification report + confusion matrix"

gh issue create --repo $REPO \
  --title "Evaluation & Comparison" \
  --label "evaluation,reporting" \
  --body "## Tasks
- [ ] Compute per-fold metrics: Accuracy, ±1 Accuracy, RMSE, Precision/Recall/F1
- [ ] Generate normalized confusion matrix
- [ ] Create comparison table:

| Encoder | Accuracy | ±1 Acc | RMSE |
|---------|----------|--------|------|
| CLIP ViT-L/14@336px | - | - | - |
| SigLIP2 So400m | - | - | - |

- [ ] Plot learning curve (accuracy vs training set size)

## Acceptance
Reproducible results table + plots saved to \`results/\`"

gh issue create --repo $REPO \
  --title "Documentation & Reproducibility" \
  --label "docs" \
  --body "## Tasks
- [ ] README with environment setup and usage instructions
- [ ] \`environment.yml\` or \`requirements.txt\`
- [ ] Document random seeds used
- [ ] Add sample results to README

## Acceptance
New contributor can reproduce results from README alone"

echo "All issues created!"