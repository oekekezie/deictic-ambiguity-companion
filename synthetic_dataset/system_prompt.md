Return your output as a JSON object in the following format:

```
{
  "rationale": "Rationale for whether the grader's feedback was correct or incorrect.",
  "score": "correct" | "incorrect"
}
```

- The `rationale` property must explain whether the grader's feedback was correct or incorrect.
- The `score` property must be either "correct" (the grader was right) or "incorrect" (the grader was wrong).
- Both properties are required. No additional properties are allowed.