from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from kds import __release__, __release_version__, __version__
from kds.cli import main
from kds.serving.api import create_app


def test_release_version_is_consistent_across_package_metadata() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert __version__ == "0.1.0"
    assert metadata["project"]["version"] == __version__
    assert __release__ == "KazDeepScan v1.0 Research"
    assert __release_version__ == "1.0.0-research"


def test_cli_reports_research_release_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "kds 0.1.0 (KazDeepScan v1.0 Research)"


def test_openapi_reports_same_release_version() -> None:
    specification = create_app().openapi()

    assert specification["info"]["title"] == "KazDeepScan Research API"
    assert specification["info"]["version"] == __release_version__
