"""Bounded in-memory multipart parsing; close buffers on success and failure."""
from fastapi import HTTPException, Request
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_AUDIO_BYTES + 64 * 1024


class MemoryAudioParser(MultiPartParser):
    # Starlette renamed this setting. Both are per-parser subclass attributes,
    # never a global override affecting unrelated uploads.
    max_file_size = MAX_REQUEST_BYTES + 1
    spool_max_size = MAX_REQUEST_BYTES + 1


async def read_transient_audio(request: Request):
    async def bounded_stream():
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                raise MultiPartException("Audio upload exceeds limit")
            yield chunk

    parser = MemoryAudioParser(request.headers, bounded_stream(), max_files=1, max_fields=0)
    form = None
    try:
        form = await parser.parse()
        audio = form.get("audio")
        if not isinstance(audio, UploadFile):
            raise HTTPException(422, "An audio file is required")
        content_type = (audio.content_type or "").lower()
        if not (content_type.startswith("audio/") or content_type == "video/webm"):
            raise HTTPException(415, "Please upload an audio recording")
        content = await audio.read(MAX_AUDIO_BYTES + 1)
        if not content:
            raise HTTPException(400, "The recording was empty")
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "The recording is too large")
        return content, audio.filename, content_type
    except MultiPartException:
        raise HTTPException(413, "Invalid or oversized audio upload") from None
    finally:
        if form is not None:
            await form.close()
        # Also cover disconnects/cancellation while a multipart file is open.
        for file in parser._files_to_close_on_error:
            file.close()
