"""``python -m attendance <command>``"""
from __future__ import annotations

import json
from pathlib import Path

import click

from .config import ROOT, load_settings


@click.group()
def cli():
    """Face-recognition attendance service."""


@cli.command()
def models():
    """Download the pinned YuNet and SFace ONNX models and verify their SHA-256."""
    from .models import ensure

    for p in ensure(load_settings()):
        click.echo(f"ok {p}")


@cli.command()
@click.option("--out", default=str(ROOT / "data"))
def demo(out):
    """Model plumbing check + one synthetic day through the API (writes data/demo.json)."""
    from .demo import run

    run(load_settings(), Path(out))
    click.echo(f"wrote {Path(out) / 'demo.json'}")


@cli.command()
@click.option("--demo-json", default=str(ROOT / "data" / "demo.json"))
@click.option("--out", default=str(ROOT / "docs" / "images"))
def report(demo_json, out):
    """Charts for the README from the demo run."""
    from .report import render_all

    for p in render_all(Path(demo_json), Path(out)):
        click.echo(f"wrote {p}")


@cli.command()
@click.option("--demo-json", default=str(ROOT / "data" / "demo.json"))
def summary(demo_json):
    """The numbers quoted in the README."""
    from .report import summary as s

    click.echo(json.dumps(s(json.loads(Path(demo_json).read_text())), indent=2))


@cli.command()
@click.option("--out", default=str(ROOT / "reports"))
def evaluate(out):
    """Accuracy on the Olivetti research faces (downloads them via scikit-learn)."""
    from .evaluate import run

    click.echo(json.dumps(run(load_settings(), Path(out)), indent=2))


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
def serve(host, port):
    """Run the API with uvicorn (keys from ATTENDANCE_ADMIN_KEY / ATTENDANCE_DEVICE_KEY)."""
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(), host=host, port=port, log_level="info")


def main():
    cli()


if __name__ == "__main__":
    main()
