"""Repeated cancellation must not release decode capacity before the child is reaped."""

import asyncio
import time
from pathlib import Path

import pytest

from app.audio import AudioIngress
from app.observability import Metrics
from tests.fake_engine import config


def test_repeated_cancellation_waits_for_reaping(monkeypatch):
    async def main():
        started = asyncio.Event()
        killed = asyncio.Event()
        waiting = asyncio.Event()
        reaped = asyncio.Event()

        class Pipe:
            async def read(self, n):
                started.set()
                await asyncio.Event().wait()

        class Process:
            returncode = None
            stdout = Pipe()
            stderr = Pipe()

            def kill(self):
                killed.set()

            async def wait(self):
                waiting.set()
                await reaped.wait()
                self.returncode = -9
                return -9

        async def spawn(*args, **kwargs):
            return Process()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        ingress = AudioIngress(config(max_concurrent_decodes=1), Metrics(), ffmpeg="/fake")
        task = asyncio.create_task(ingress.decode(Path("/unused"), time.monotonic() + 10))
        await started.wait()
        task.cancel()
        await killed.wait()
        await waiting.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert ingress._decode_slots.locked()
        assert ingress._decoding == 1
        reaped.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not ingress._decode_slots.locked()
        assert ingress._decoding == 0

    asyncio.run(main())
