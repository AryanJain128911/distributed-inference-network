"""Exercise the terminal roles, model-consent boundary, and streamed client contract."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from inference_network.client import stream_request
from inference_network.config import CoordinatorSettings
from inference_network.errors import NetworkError
from inference_network.launcher import ServiceProcess
from inference_network.model_setup import prepare_model, snapshot_complete
from inference_network.schemas import ChatRequest


def frames(simulated=False):
    base = {
        "id": "chat-test",
        "created": 1,
        "model": "test-model",
        "metadata": {"provider_id": "studio", "simulated": simulated},
        "usage": None,
    }
    choices = [
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None},
        {"index": 0, "delta": {"content": "Hello "}, "finish_reason": None},
        {"index": 0, "delta": {"content": "world"}, "finish_reason": None},
        {"index": 0, "delta": {}, "finish_reason": "stop"},
    ]
    chunks = [{**base, "choices": [choice]} for choice in choices]
    chunks.append(
        {
            **base,
            "choices": [],
            "usage": None
            if simulated
            else {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
    )
    return [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks] + ["data: [DONE]\n\n"]


async def request_frames(parts, callback=None):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content="".join(parts).encode()
        )
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        return await stream_request(
            client,
            ChatRequest.model_validate(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                }
            ),
            callback,
        )


@pytest.mark.parametrize("simulated", [True, False])
async def test_client_streams_text_and_requires_actual_counters_for_real_output(simulated):
    observed = []
    result = await request_frames(frames(simulated), observed.append)
    assert observed == ["Hello ", "world"]
    assert result.content == "Hello world"
    assert result.provider_id == "studio"
    assert result.simulated is simulated
    assert (result.usage is None) is simulated
    assert result.first_text_seconds <= result.elapsed_seconds
    if not simulated:
        assert result.usage.total_tokens == 7


@pytest.mark.parametrize("mutation", ["truncated", "identity", "order", "error", "after_done"])
async def test_client_does_not_report_success_for_failed_or_malformed_stream(mutation):
    parts = frames()
    if mutation == "truncated":
        parts.pop()
    elif mutation == "identity":
        parts[2] = parts[2].replace('"studio"', '"imposter"')
    elif mutation == "order":
        parts[3], parts[4] = parts[4], parts[3]
    elif mutation == "error":
        parts[-1] = 'event: error\ndata: {"secret": "never echo"}\n\n'
    else:
        parts.append(parts[0])
    with pytest.raises((ValueError, NetworkError)) as failure:
        await request_frames(parts)
    assert "never echo" not in str(failure.value)


async def test_client_preserves_bounded_admission_error_without_echoing_remote_body():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            503, json={"error": {"code": "no_eligible_provider", "message": "secret"}}
        )
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        with pytest.raises(NetworkError) as failure:
            await stream_request(
                client,
                ChatRequest.model_validate(
                    {
                        "model": "test",
                        "messages": [{"role": "user", "content": "Hi"}],
                    }
                ),
            )
    assert failure.value.code == "no_eligible_provider"
    assert "secret" not in str(failure.value)


def make_snapshot(path: Path):
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (path / filename).write_text("synthetic test file")


def test_partial_cached_snapshot_is_not_mistaken_for_downloaded_weights(tmp_path):
    make_snapshot(tmp_path)
    assert snapshot_complete(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    assert not snapshot_complete(tmp_path)


def test_malformed_weight_index_is_treated_as_incomplete_cache(tmp_path):
    make_snapshot(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": []}')
    assert not snapshot_complete(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}})
    )
    assert not snapshot_complete(tmp_path)


@pytest.mark.parametrize("answer,downloads", [("n", 0), ("", 0), ("yes", 1)])
def test_missing_model_download_requires_explicit_consent(monkeypatch, answer, downloads):
    calls = []

    class MissingCache(Exception):
        pass

    def snapshot(*args, **kwargs):
        if kwargs.get("local_files_only"):
            raise MissingCache()
        calls.append(kwargs)

    monkeypatch.setattr("inference_network.model_setup.check_mlx_support", lambda: None)
    monkeypatch.setattr(
        "inference_network.model_setup.importlib.import_module",
        lambda name: (
            SimpleNamespace(snapshot_download=snapshot)
            if name == "huggingface_hub"
            else SimpleNamespace(LocalEntryNotFoundError=MissingCache)
        ),
    )
    if downloads:
        prepare_model(lambda question: answer)
        assert calls[0]["token"] is False
        assert calls[0]["revision"] == "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
    else:
        with pytest.raises(RuntimeError, match="declined"):
            prepare_model(lambda question: answer)
    assert len(calls) == downloads


def test_complete_cache_never_prompts_or_downloads(monkeypatch, tmp_path):
    make_snapshot(tmp_path)
    calls = []

    def snapshot(*args, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("inference_network.model_setup.check_mlx_support", lambda: None)
    monkeypatch.setattr(
        "inference_network.model_setup.importlib.import_module",
        lambda name: (
            SimpleNamespace(snapshot_download=snapshot)
            if name == "huggingface_hub"
            else SimpleNamespace(LocalEntryNotFoundError=RuntimeError)
        ),
    )
    prepare_model(lambda question: pytest.fail("Cached model must not prompt"))
    assert calls == [
        {
            "revision": "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
            "local_files_only": True,
            "token": False,
        }
    ]


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.mark.integration
def test_sender_create_owns_and_stops_coordinator_without_loading_model():
    port = free_port()
    result = subprocess.run(
        [sys.executable, "-m", "inference_network", "interactive"],
        input=f"1\n1\n{port}\n/quit\n",
        text=True,
        capture_output=True,
        timeout=15,
        env={key: value for key, value in os.environ.items() if not key.startswith("DIN_")},
    )
    assert result.returncode == 0, result.stderr
    assert "None yet" in result.stdout
    assert "Download" not in result.stdout
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))


@pytest.mark.integration
def test_join_sender_leaves_existing_coordinator_running():
    settings = CoordinatorSettings(port=free_port())
    with ServiceProcess("coordinator", settings) as service:
        result = subprocess.run(
            [sys.executable, "-m", "inference_network", "interactive"],
            input=f"1\n2\n{service.url}\n/quit\n",
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        service.check_running()
        with httpx.Client(trust_env=False) as client:
            assert client.get(f"{service.url}/health").status_code == 200
