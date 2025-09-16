#!/usr/bin/env python3
import argparse
import os
import sys

START_MARKER = "<!-- TREE:START -->"
END_MARKER = "<!-- TREE:END -->"


def generate_tree(root="."):
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip .git and venvs
        parts = dirpath.split(os.sep)
        if ".git" in parts or ".venv" in parts:
            continue
        depth = len(parts) - len(root.split(os.sep))
        if depth < 0:
            depth = 0
        indent = "  " * depth
        name = (
            os.path.basename(dirpath)
            if dirpath != root
            else os.path.basename(os.path.abspath(root))
        )
        if dirpath == root:
            lines.append(f"{name}/")
        else:
            lines.append(f"{indent}└─ {name}/")
        for f in sorted(filenames):
            if f.startswith("."):
                continue
            lines.append(f"{indent}   └─ {f}")
    return "```\n" + "\n".join(lines) + "\n```"


def update_readme(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    tree_md = generate_tree(".")
    if START_MARKER in content and END_MARKER in content:
        prefix, rest = content.split(START_MARKER, 1)
        old, suffix = rest.split(END_MARKER, 1)
        new = prefix + START_MARKER + "\n" + tree_md + "\n" + END_MARKER + suffix
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new)
        print("README updated.")
    else:
        print("Markers not found in README; printing tree:\n")
        print(tree_md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument(
        "--inplace", action="store_true", help="Update README between TREE markers"
    )
    ap.add_argument(
        "--print-only", action="store_true", help="Print tree to stdout only"
    )
    args = ap.parse_args()
    if args.print_only:
        print(generate_tree("."))
        sys.exit(0)
    if args.inplace:
        update_readme(args.readme)
    else:
        print(generate_tree("."))
