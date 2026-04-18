import os

def read_range_thread(fd: int, start: int, end: int) -> str:
    return os.pread(fd, end - start, start).decode(encoding="utf-8", errors="ignore")


def read_range_process(path: str, start: int, end: int) -> str:
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(end - start).decode("utf-8", "ignore")
