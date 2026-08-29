# Results

Only two result classes are kept:

```text
results/
├── final/          Concise performance summaries produced after Formal runs
└── intermediate/   Search studies, trial runs, probes, and profiles
```

- `final/`: concise results intentionally selected for submission.
- `intermediate/`: disposable run JSON, profiles, and one resumable Optuna
  SQLite database.

Old Policy-based results are not reused by the generated-configuration runtime.
