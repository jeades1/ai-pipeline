import zipfile
from pathlib import Path

# Output zip
out_zip = Path("ai_pipeline_review.zip")

# Directories/files to exclude
exclude_dirs = {
    ".git",
    "data",
    "artifacts",
    "__pycache__",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".vscode",
}
exclude_exts = {".pyc", ".pyo", ".pyd", ".so", ".dll"}
exclude_files = {"ai_pipeline_review.zip"}  # don’t zip itself


def should_include(path: Path) -> bool:
    parts = set(path.parts)
    if parts & exclude_dirs:
        return False
    if path.suffix in exclude_exts:
        return False
    if path.name in exclude_files:
        return False
    return True


with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in Path(".").rglob("*"):
        if p.is_file() and should_include(p):
            zf.write(p, p.relative_to("."))
            print("Added:", p)

print(f"\n[done] Wrote {out_zip}")
