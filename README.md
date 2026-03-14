# Warehouse Scanner

## Google Vision Credentials

Google Vision credentials must be provided through one of these options:

- Google Secret Manager
- The `GOOGLE_APPLICATION_CREDENTIALS` environment variable

The API defaults to looking for a mounted secret at `/secrets/google-vision-key` when
`GOOGLE_APPLICATION_CREDENTIALS` is not set.

Do not commit service account JSON files. Keep credentials out of the repository and
out of Git history.

Example PowerShell setup:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\secure\google-vision-key.json"
```

Example Linux/macOS setup:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/secure/google-vision-key.json
```

## Git Safety

This repository blocks common secret paths and markers with the local pre-commit hook
in `.githooks/pre-commit`.

Enable it once per clone with:

```bash
git config core.hooksPath .githooks
```

The hook blocks commits that include:

- `.json` files
- files under `keys/`
- the strings `service_account` or `google_vision_key`

If GitHub push protection still reports an old secret in history, rewrite the affected
path before pushing. Example:

```bash
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch api/keys/google_vision_key.json keys/google_vision_key.json" \
  --prune-empty --tag-name-filter cat -- --all
git push --force-with-lease --all
git push --force-with-lease --tags
```

Rotate any credential that was ever committed, even if it has since been deleted.
