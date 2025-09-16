# main.py — single entry point that forwards to the unified Typer CLI
from tools.cli.cli import app

if __name__ == "__main__":
    app()
