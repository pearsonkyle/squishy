from __future__ import annotations

from types import SimpleNamespace

import squishy.cli as cli




async def test_user_configured_model_via_args():
    args = SimpleNamespace(model="my-model")
    assert cli._user_configured_model(args) is True


async def test_user_configured_model_via_env(monkeypatch):
    monkeypatch.setenv("SQUISHY_MODEL", "env-model")
    args = SimpleNamespace(model=None)
    assert cli._user_configured_model(args) is True


async def test_user_configured_model_default(monkeypatch):
    monkeypatch.delenv("SQUISHY_MODEL", raising=False)
    args = SimpleNamespace(model=None)
    assert cli._user_configured_model(args) is False






