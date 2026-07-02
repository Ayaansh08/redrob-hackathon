# Redrob Candidate Ranker

Run the full main-dataset pipeline from PowerShell:

```powershell
python scripts/01_preprocess.py; if ($LASTEXITCODE -eq 0) { python scripts/02_embed.py }; if ($LASTEXITCODE -eq 0) { python scripts/03_score.py }; if ($LASTEXITCODE -eq 0) { python scripts/04_generate_submission.py }
```
