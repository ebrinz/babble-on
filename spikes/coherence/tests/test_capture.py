from capture import capture_to


def test_capture_to_writes_requested_total(tmp_path):
    out = tmp_path / "c.bin"
    assert capture_to(lambda n: b"\x01" * n, out, 1000, chunk=300) == 1000
    assert out.stat().st_size == 1000


def test_capture_to_keeps_partial_file_on_dropout(tmp_path, capsys):
    calls = []

    def read(n):
        if len(calls) == 2:
            raise OSError("Device not configured")
        calls.append(n)
        return b"\x02" * n

    out = tmp_path / "c.bin"
    assert capture_to(read, out, 1000, chunk=300) == 600
    assert out.stat().st_size == 600
    assert "600 bytes" in capsys.readouterr().err
