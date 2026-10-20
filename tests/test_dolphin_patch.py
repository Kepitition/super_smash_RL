import pytest

from smash_rl import dolphin

# Shape of the real exe: the pipe name literal and, elsewhere, the inlined "bytes_read == 0 -> return".
EXE = (b"\x00" * 16 + b"\\\\.\\pipe\\slippibot\x00\x00\x00\x00\x00\x00slippibot\x00"
       + b"\x90" * 8 + dolphin.EARLY_RETURN + b"\x90" * 8)


def test_patch_renames_the_pipes_and_removes_the_early_return():
    patched = dolphin.patch_dolphin_exe(EXE, b"slip51441")
    assert len(patched) == len(EXE)
    assert b"\\\\.\\pipe\\slip51441\x00" in patched and b"\\\\.\\pipe\\slippibot" not in patched
    assert b"slippibot\x00" in patched  # the UI device name stays
    assert dolphin.EARLY_RETURN not in patched and dolphin.WAIT_FOR_FLUSH in patched


def test_patch_refuses_an_unknown_build():
    with pytest.raises(RuntimeError, match="early return"):
        dolphin.patch_dolphin_exe(EXE.replace(dolphin.EARLY_RETURN, b"\x90" * len(dolphin.EARLY_RETURN)), b"slip51441")
    with pytest.raises(RuntimeError, match="pipes"):
        dolphin.patch_dolphin_exe(EXE.replace(b"slippibot\x00\x00", b"otherbots\x00\x00"), b"slip51441")
