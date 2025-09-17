# tools/cli/cli.py — unified command-line interface
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import typer

app = typer.Typer(help="AI Biomarker Discovery Pipeline — Unified CLI")


# ---------- Helpers ----------
def _py() -> str:
    return sys.executable or "python3"


def _run(cmd: list[str]) -> None:
    typer.echo(f"[cli] $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _exists(p: str | Path) -> bool:
    return Path(p).exists()


# (demo commands removed)
@app.command("first-win")
def first_win(
    disease: str = typer.Option(
        "cardiovascular", help="Benchmark panel: cardiovascular|oncology|default|aki"
    ),
    k: int = typer.Option(20, help="Top-K for precision/recall"),
    top_trace: int = typer.Option(5, help="How many top genes to trace in the report"),
):
    """Run the end-to-end 'first win' demo: rerank with priors+causal, benchmark, and audit."""
    _run(
        [
            _py(),
            "-m",
            "tools.bench.first_win",
            "--disease",
            str(disease),
            "--k",
            str(k),
            "--top-trace",
            str(top_trace),
        ]
    )


@app.command("build-priors")
def build_priors(
    context: str = typer.Option(
        "kidney", help="Context for priors (kidney defaults to kidney/cortex/medulla)"
    ),
    outdir: str = typer.Option("data/processed/priors", help="Where to write priors"),
    min_score: float = typer.Option(
        0.0, help="Minimum score filter for OpenTargets priors"
    ),
):
    """Build unified priors (OpenTargets + GTEx + HPA) if their source files exist."""
    _run(
        [
            _py(),
            "-m",
            "tools.priors.build_all",
            "--outdir",
            str(outdir),
            "--context",
            str(context),
            "--min-score",
            str(min_score),
        ]
    )


@app.command("build-visuals")
def build_visuals():
    """Regenerate conceptual and competitive visuals (disease-agnostic)."""
    _run([_py(), "-m", "tools.plots.enhanced_visuals"])
    _run([_py(), "-m", "tools.plots.competitive_report"])


# (demo UI removed)


# ---------- Kept utilities from your repo ----------
@app.command("render-diagrams")
def render_diagrams():
    """
    Render Mermaid diagrams using your existing script or Makefile target.
    """
    if _exists("render_diagrams_make.sh"):
        _run(["bash", "render_diagrams_make.sh"])
    elif _exists("render_mermaid.sh"):
        _run(["bash", "render_mermaid.sh"])
    else:
        typer.echo("No render script found. Trying `make render-diagrams`…")
        _run(["make", "render-diagrams"])


@app.command("tissue-chip-demo")
def tissue_chip_demo():
    """Run advanced tissue-chip integration demo with multicellular architecture."""
    if _exists("test_tissue_chip_integration.py"):
        _run([_py(), "test_tissue_chip_integration.py"])
    else:
        typer.echo("test_tissue_chip_integration.py not found; skipping.", err=True)


@app.command("update-readme-tree")
def update_readme_tree():
    """
    Refresh README file tree snippet using your existing utility if present.
    """
    if _exists("update_repo_tree.py"):
        _run([_py(), "update_repo_tree.py"])
    else:
        typer.echo("update_repo_tree.py not found; skipping.", err=True)


# Optional: preserve specialized utilities if they exist in your repo
@app.command("modules-from-geo")
def modules_from_geo(
    dir: str = typer.Argument(None),
    modules_json: str = typer.Option("modeling/modules/tubular_modules_v1.json"),
):
    """
    (Optional) Your existing module-updater; will only run if scripts are present.
    """
    script = Path("scripts") / "make_modules_from_geo.py"
    if not _exists(script):
        typer.echo(f"{script} not found; command unavailable.", err=True)
        raise typer.Exit(code=1)
    _run([_py(), str(script), dir, "--modules-json", modules_json])


# (MIMIC-specific wrapper removed)

# ---------- Deprecated orchestration ----------
# (deprecated orchestration removed)


# ---------- Entry ----------
def app_main():
    app()


if __name__ == "__main__":
    app()
