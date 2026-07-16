import json

import pytest

from modal_app import api, worker


def _queued_status(run_id: str) -> dict:
    return {
        "runId": run_id,
        "state": "queued",
        "createdAt": "2026-07-14T12:00:00Z",
        "startedAt": None,
        "completedAt": None,
        "error": None,
        "numAuthorLabels": 2,
        "numAlgorithms": 1,
        "numRows": None,
        "resultCsvPath": None,
        "resultJsonPath": None,
    }


class _AsyncReload:
    def __init__(self, on_reload):
        self.calls = 0
        self.on_reload = on_reload

    async def aio(self):
        self.calls += 1
        self.on_reload(self.calls)


class _AsyncVolume:
    def __init__(self, on_reload):
        self.reload = _AsyncReload(on_reload)


class _SyncVolume:
    def __init__(self):
        self.commit_calls = 0
        self.reload_calls = 0

    def commit(self):
        self.commit_calls += 1

    def reload(self):
        self.reload_calls += 1


async def test_worker_reloads_until_initial_status_is_visible(tmp_path, monkeypatch):
    run_id = "run-test"
    monkeypatch.setattr(worker.app_config, "REMOTE_USER_DIR", str(tmp_path))
    monkeypatch.setattr(worker, "_STATUS_LOAD_RETRY_SECONDS", 0)

    def publish_status(reload_count: int) -> None:
        if reload_count != 2:
            return
        path = tmp_path / run_id / "status.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_queued_status(run_id)))

    volume = _AsyncVolume(publish_status)

    status = await worker._load_initial_status(run_id, volume)

    assert status == _queued_status(run_id)
    assert volume.reload.calls == 2


async def test_worker_rejects_incomplete_initial_status(tmp_path, monkeypatch):
    run_id = "run-test"
    path = tmp_path / run_id / "status.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"runId": run_id}))

    monkeypatch.setattr(worker.app_config, "REMOTE_USER_DIR", str(tmp_path))
    monkeypatch.setattr(worker, "_STATUS_LOAD_ATTEMPTS", 2)
    monkeypatch.setattr(worker, "_STATUS_LOAD_RETRY_SECONDS", 0)
    volume = _AsyncVolume(lambda _: None)

    with pytest.raises(RuntimeError, match="missing or incomplete"):
        await worker._load_initial_status(run_id, volume)


def test_api_status_write_is_atomic_and_committed(tmp_path, monkeypatch):
    run_id = "run-test"
    status = _queued_status(run_id)
    volume = _SyncVolume()
    monkeypatch.setattr(api.app_config, "REMOTE_USER_DIR", str(tmp_path))

    api._write_status(run_id, status, volume)

    assert api._read_status(run_id, volume) == status
    assert volume.commit_calls == 1
    assert volume.reload_calls == 1
    assert not (tmp_path / run_id / ".status.json.tmp").exists()
