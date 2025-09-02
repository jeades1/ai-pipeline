import typer
app = typer.Typer(help="ai-pipeline CLI")

@app.command()
def hello(name: str = "world"):
    print(f"hello, {name}")

if __name__ == "__main__":
    app()