"""One Slippi Dolphin process driven through libmelee, with the fixes Windows needs.

`Dolphin` starts and stops the emulator and hands out game states; the per-port
copies, buffered controller, robust connect/stop and the job object below are
what make many Dolphins at once work on Windows.
"""

import configparser
import os
import platform
import shutil
import time

import melee
from melee.slippstream import SlippstreamClient

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM = platform.system()
if SYSTEM == "Darwin":
    DEFAULT_DOLPHIN = os.path.join(PROJECT_DIR, "dolphin", "Slippi Dolphin.app", "Contents", "MacOS", "Slippi Dolphin")
elif SYSTEM == "Windows":  # the portable FM-Slippi zip, unpacked by setup.ps1
    DEFAULT_DOLPHIN = os.path.join(PROJECT_DIR, "dolphin", "Slippi Dolphin", "Slippi Dolphin.exe")
else:  # Linux: libmelee runs the Slippi Online AppImage directly
    DEFAULT_DOLPHIN = os.path.join(PROJECT_DIR, "dolphin", "Slippi_Online-x86_64.AppImage")
HEADLESS_DOLPHIN = os.path.join(PROJECT_DIR, "dolphin", "Slippi_Online-x86_64-ExiAI.AppImage")
# The ExiAI build (Null video, fast-forward) only exists as a Linux AppImage.
CAN_RUN_HEADLESS = SYSTEM == "Linux"
DEFAULT_ISO = os.path.join(PROJECT_DIR, "Super Smash Bros Melee v1.02.iso")
# Per-port Dolphin copies with their own controller pipe names (Windows), see _windows_dolphin_copy.
INSTANCES_DIR = os.path.join(PROJECT_DIR, "dolphin", "instances")
WINDOWS_PIPE = b"\\\\.\\pipe\\slippibot"  # hard-coded in the Windows build, followed by the port 1..4
# PipeDevice::UpdateInput in the Windows build of Slippi Dolphin 3.6.4 returns as soon as a
# read finds no data (`if (bytes_read == 0) return;`), so with blocking pipes it still runs
# the frame on the old input whenever the bot's input is a moment late, and drifts ahead of
# the bot. These bytes are that check (mov eax,[rbp+48h]; test eax,eax; je <return>; jle
# <parse>); a 6-byte NOP in place of the je makes "no data yet" fall through to parsing and
# the wait-for-FLUSH loop, as in mainline Slippi Dolphin.
EARLY_RETURN = bytes.fromhex("8b454885c0" "0f84e4020000" "0f8ef5000000")
WAIT_FOR_FLUSH = bytes.fromhex("8b454885c0" "660f1f440000" "0f8ef5000000")
# Bump when the patches change, so existing per-port copies are rebuilt.
PATCH_VERSION = "2"
# With truly blocking input, Dolphin sometimes wants an input on a frame that sends no game
# event (in menus and loading), while libmelee only sends one after an event: a deadlock.
# Polling instead re-sends the input whenever no frame came for this long. In a game,
# frames come within milliseconds, so it never fires there.
STALL_SECONDS = 1.0


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:  # e.g. a file system without hard links
        shutil.copy2(src, dst)


def patch_dolphin_exe(exe: bytes, tag: bytes) -> bytes:
    """The Windows Dolphin exe with its pipes renamed to \\\\.\\pipe\\<tag>N and truly blocking input.

    `tag` must be as long as "slippibot" (9 bytes) so nothing else in the exe moves.
    """
    if exe.count(WINDOWS_PIPE) != 1:
        raise RuntimeError(f"this Dolphin doesn't name its pipes {WINDOWS_PIPE.decode()}N once")
    if exe.count(EARLY_RETURN) != 1:
        raise RuntimeError("this Dolphin build doesn't have the known pipe early return to patch "
                           "(expected Slippi Dolphin 3.6.4 for Windows)")
    exe = exe.replace(WINDOWS_PIPE, WINDOWS_PIPE[:-len(tag)] + tag)
    return exe.replace(EARLY_RETURN, WAIT_FOR_FLUSH)


def _windows_dolphin_copy(dolphin_exe: str, port: int) -> tuple:
    """(exe, pipe prefix) of a copy of Dolphin whose controller pipes belong to `port` alone.

    The Windows build always names its input pipes \\\\.\\pipe\\slippibot1..4, and those
    names are global, so a second Dolphin can't create them and its bot would drive the
    first one. Each Slippi port gets its own copy (hard links, except the exe) whose exe
    says \\\\.\\pipe\\slip<port> instead and waits for the bot's input (patch_dolphin_exe).
    """
    tag = f"slip{port:05d}".encode()
    dst_dir = os.path.join(INSTANCES_DIR, tag.decode())
    dst_exe = os.path.join(dst_dir, os.path.basename(dolphin_exe))
    marker = os.path.join(dst_dir, "smash_rl_patch.txt")
    current = os.path.isfile(marker) and open(marker).read() == PATCH_VERSION
    if not current or os.path.getmtime(dst_exe) < os.path.getmtime(dolphin_exe):
        with open(dolphin_exe, "rb") as f:
            exe = patch_dolphin_exe(f.read(), tag)
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)  # fails if a Dolphin still runs from it
        shutil.copytree(os.path.dirname(dolphin_exe), dst_dir, copy_function=_link_or_copy)
        os.remove(dst_exe)  # a hard link to the original: replace it rather than write through it
        with open(dst_exe, "wb") as f:
            f.write(exe)
        with open(marker, "w") as f:
            f.write(PATCH_VERSION)
    return dst_exe, (WINDOWS_PIPE[:-len(tag)] + tag).decode()


class _Controller(melee.Controller):
    """libmelee Controller that sends an input all at once, on flush().

    On Windows libmelee writes each command to the pipe as soon as it is made, and Dolphin
    applies whatever has arrived when it polls: the neutral stick of the end-of-step
    release_all() got into the game as frames without input, more of them the busier the
    CPU. On Linux and macOS the pipe is a buffered file that only flush() sends; this copies that.
    """

    def __init__(self, *args, **kwargs):
        self._pending = []
        super().__init__(*args, **kwargs)

    def _write(self, command):
        if SYSTEM == "Windows":
            self._pending.append(command)
        else:
            super()._write(command)

    def flush(self):
        super().flush()  # queues FLUSH
        if self._pending:
            data, self._pending = "".join(self._pending), []
            super()._write(data)


class _Console(melee.Console):
    """libmelee Console that can point controllers at renamed Windows pipes and always stops."""

    def __init__(self, *args, pipe_prefix: str = None, **kwargs):
        self.pipe_prefix = pipe_prefix
        super().__init__(*args, **kwargs)

    def get_dolphin_pipes_path(self, port):
        if self.pipe_prefix:
            return self.pipe_prefix + str(port)
        return super().get_dolphin_pipes_path(port)

    def connect(self):
        # libmelee drops its network process after a failed attempt (each waits 10 s for
        # Dolphin), so without a new client every retry would crash instead of waiting longer.
        if self._slippstream._worker is None:
            self._slippstream = SlippstreamClient(self.slippi_address, self.slippi_port)
        return super().connect()

    def stop(self):
        # libmelee waits for its network process to exit before killing Dolphin, but that
        # process can be stuck writing game states nobody reads any more (a Windows pipe
        # holds only a few KB), which hung env.close(). Kill Dolphin first, then give the
        # network process a moment to notice before killing it too.
        if self._process is not None:
            self._process.kill()
            self._process.wait()
            self._process = None
        stream = self._slippstream
        worker = getattr(stream, "_worker", None)
        if worker is not None and worker.pid is not None:
            stream._shutdown.set()
            worker.join(timeout=2)
            if worker.is_alive():
                worker.kill()
                worker.join()
            stream._buffer.close()
            stream._worker = None
        super().stop()


_JOB = None


def kill_children_on_exit():
    """Windows: every process started from now on (and theirs) dies with this one, however it exits.

    Linux relies on signals reaching the workers to close Dolphin; on Windows a closed
    terminal or a killed process would otherwise leave workers and Dolphin windows behind.
    This process joins a job that only it holds a handle to; children join it automatically,
    and Windows kills the whole job when the handle closes.
    """
    global _JOB
    if SYSTEM != "Windows" or _JOB is not None:
        return
    import win32api
    import win32job

    _JOB = win32job.CreateJobObject(None, "")
    info = win32job.QueryInformationJobObject(_JOB, win32job.JobObjectExtendedLimitInformation)
    info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    win32job.SetInformationJobObject(_JOB, win32job.JobObjectExtendedLimitInformation, info)
    win32job.AssignProcessToJobObject(_JOB, win32api.GetCurrentProcess())


class Dolphin:
    """One Dolphin: `pad` drives port 1, `cpu_pad` port 2 (None when a person plays port 2).

    save_replays: a directory where Dolphin writes a Slippi replay of every game, or None.
    Each port writes into its own subfolder (port<port>): Slippi names replays by the
    second, so parallel games ending together overwrote each other.
    """

    def __init__(self, port: int, *, headless: bool = False, dolphin_path: str = None,
                 dolphin_home: str = None, iso_path: str = DEFAULT_ISO, gfx_backend: str = "OGL",
                 gfx_settings: dict = None, emulation_speed: float = 0.0, connect_timeout: float = 120.0,
                 human_opponent: bool = False, instant_restart: bool = False, save_replays: str = None,
                 window_size: tuple = None):
        if headless and not CAN_RUN_HEADLESS:
            raise ValueError(f"headless mode needs Linux (the ExiAI Dolphin is a Linux AppImage), not {SYSTEM}")
        self.port = port
        self.headless = headless
        self.dolphin_path = dolphin_path or (HEADLESS_DOLPHIN if headless else DEFAULT_DOLPHIN)
        self.dolphin_home = dolphin_home  # None: a fresh temporary Dolphin user folder per start
        self.iso_path = iso_path
        self.gfx_backend = gfx_backend
        self.gfx_settings = gfx_settings or {}  # {"Section": {"Key": value}} for Dolphin's GFX.ini
        self.emulation_speed = emulation_speed
        self.connect_timeout = connect_timeout
        self.human_opponent = human_opponent
        # Small windows for training; a person needs a big one.
        self.window_size = window_size or ((1152, 864) if human_opponent else (480, 360))
        self.instant_restart = instant_restart
        self.save_replays = os.path.join(save_replays, f"port{port}") if save_replays else None
        for path in (self.dolphin_path, self.iso_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} not found - see setup.ps1 (Windows) or setup.sh")
        self.console = self.pad = self.cpu_pad = None

    def start(self):
        self.stop()
        kill_children_on_exit()
        if self.save_replays:
            os.makedirs(self.save_replays, exist_ok=True)
        dolphin_path, pipe_prefix = self.dolphin_path, None
        if SYSTEM == "Windows":
            dolphin_path, pipe_prefix = _windows_dolphin_copy(self.dolphin_path, self.port)
        console = _Console(
            pipe_prefix=pipe_prefix,
            path=dolphin_path,
            dolphin_home_path=self.dolphin_home,
            tmp_home_directory=self.dolphin_home is None,
            slippi_port=self.port,
            # Blocking input makes Dolphin wait for the agent every frame, which would freeze
            # a pause by a person (no game states arrive to answer with while paused).
            blocking_input=not self.human_opponent,
            polling_mode=True,
            polling_timeout=STALL_SECONDS,
            fullscreen=False,
            disable_audio=not self.human_opponent,
            save_replays=bool(self.save_replays),
            replay_dir=self.save_replays,
            gfx_backend="Null" if self.headless else self.gfx_backend,
            log_types=[],
            instant_match_restart=self.instant_restart,
            use_exi_inputs=self.headless,
            enable_ffw=self.headless,
        )
        game_ini = os.path.join(console._get_dolphin_home_path(), "GameSettings", "GALE01r2.ini")
        with open(game_ini) as f:
            text = f.read()
        # Instant Match restarts the same match as soon as one ends, skipping the menus.
        # Its third line selects a random stage; 00000000 keeps the stage we picked.
        head, sep, tail = text.partition("$Optional: Instant Match")
        tail = tail.replace("00000001 7C0802A6", "00000000 7C0802A6", 1)
        # This Slippi build only reads EmulationSpeed from the per-game ini, not Dolphin.ini.
        with open(game_ini, "w") as f:
            f.write(head + sep + tail + f"\n[Core]\nEmulationSpeed = {self.emulation_speed}\n")
        dolphin_ini = os.path.join(console._get_dolphin_config_path(), "Dolphin.ini")
        # Lower-case keys like libmelee writes them (Dolphin ignores case): mixing both would
        # leave duplicate keys behind in a Dolphin folder that is kept between runs.
        cfg = configparser.ConfigParser()
        cfg.read(dolphin_ini)
        width, height = self.window_size
        cfg.set("Display", "RenderWindowWidth", str(width))
        cfg.set("Display", "RenderWindowHeight", str(height))
        cfg.set("Display", "RenderWindowAutoSize", "False")
        if not cfg.has_section("Interface"):
            cfg.add_section("Interface")
        cfg.set("Interface", "OnScreenDisplayMessages", "False")  # no text over the game
        if self.human_opponent:  # port 2 is a standard controller mapped in Dolphin
            cfg.set("Core", "SIDevice1", melee.ControllerType.STANDARD.value)
        with open(dolphin_ini, "w") as f:
            cfg.write(f)
        if self.gfx_settings:
            gfx_ini = os.path.join(console._get_dolphin_config_path(), "GFX.ini")
            cfg = configparser.ConfigParser()
            cfg.read(gfx_ini)
            for section, values in self.gfx_settings.items():
                if not cfg.has_section(section):
                    cfg.add_section(section)
                for key, value in values.items():
                    cfg.set(section, key, str(value))
            with open(gfx_ini, "w") as f:
                cfg.write(f)

        self.pad = _Controller(console=console, port=1, type=melee.ControllerType.STANDARD)
        self.cpu_pad = None
        if not self.human_opponent:
            self.cpu_pad = _Controller(console=console, port=2, type=melee.ControllerType.STANDARD)
        console.run(iso_path=self.iso_path)
        self.console = console
        # One Dolphin starts in a few seconds, but many starting at once can take much longer.
        deadline = time.time() + self.connect_timeout
        while not console.connect():
            if not self.alive() or time.time() > deadline:
                self.stop()
                raise RuntimeError(f"could not connect to dolphin on port {self.port}")
            time.sleep(0.5)
        for pad in (self.pad, self.cpu_pad):
            # Only Windows can fail here (a pipe that is missing or taken); libmelee just returns False.
            if pad is not None and not pad.connect():
                self.stop()
                raise RuntimeError(f"could not open the controller pipe {pad.pipe_path}")

    def stop(self):
        if self.console is not None:
            try:
                self.console.stop()
            except Exception:
                pass
            self.console = None

    def alive(self) -> bool:
        proc = getattr(self.console, "_process", None)
        return self.console is not None and proc is not None and proc.poll() is None

    def next_state(self, give_up: float = None) -> melee.GameState:
        """The next frame; each empty poll re-sends the current input (see STALL_SECONDS).

        Gives up after `give_up` seconds without a frame: 60 by default, never when a person
        plays port 2 (they may pause). A closed Dolphin window is reported at once.
        """
        if give_up is None:
            give_up = float("inf") if self.human_opponent else 60.0
        deadline = time.time() + give_up
        while True:
            gs = self.console.step()
            if gs is not None:
                return gs
            if not self.alive():
                raise RuntimeError("the Dolphin window was closed")
            if time.time() > deadline:
                raise RuntimeError("dolphin stopped sending game states")
