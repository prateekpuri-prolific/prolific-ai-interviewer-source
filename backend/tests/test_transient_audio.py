from unittest.mock import patch
from tempfile import SpooledTemporaryFile

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from app.audio.transient_upload import read_transient_audio


def request_for(content):
    body = b'--review\r\nContent-Disposition: form-data; name="audio"; filename="test.webm"\r\nContent-Type: audio/webm\r\n\r\n' + content + b'\r\n--review--\r\n'
    sent = False
    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}
    return Request({"type": "http", "headers": [(b"content-type", b"multipart/form-data; boundary=review")]}, receive)


async def test_large_audio_stays_in_memory_and_is_closed():
    files = []
    def track(*args, **kwargs):
        f = SpooledTemporaryFile(*args, **kwargs)
        files.append(f)
        return f
    with patch("starlette.formparsers.SpooledTemporaryFile", side_effect=track):
        content, name, mime = await read_transient_audio(request_for(b"x" * (2 * 1024 * 1024)))
    assert len(content) == 2 * 1024 * 1024
    assert name == "test.webm" and mime == "audio/webm"
    assert files and all(not f._rolled and f.closed for f in files)


async def test_oversized_audio_is_rejected_and_buffers_closed():
    with pytest.raises(HTTPException) as error:
        await read_transient_audio(request_for(b"x" * (26 * 1024 * 1024)))
    assert error.value.status_code == 413
