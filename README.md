# hema-footage-analyzer
Longsword fight video in, timestamped timeline of exchanges out (pose, blade and guard detection in Python).

## Status

Phase 1 (detection fixes) done: both fighters tracked on all four reference clips. See CLAUDE.MD for the full plan.

## Next

Phase 1.5: labelling helper and reference labels for the four clips.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Training footage and outputs are not in this folder (or in git); they stay in `D:\GIT\HEMA analyzer\`.
