"""One way to run a tab's long job (Seeds, Grow, Boundary Compute) in the background.

    start   the run button(s) off, Stop on; a grey "<Doing>…" status line; the progress bar
            busy until the first real progress (done / total reported by the core)
    Stop    cooperative: sets the threading.Event that the core checks between steps
            (erosion steps, threshold steps, growth iterations, chunks), which then raises
            sprout_core.Cancelled. Nothing is killed: files being written are finished.
    end     green "✓ <summary> (<t> s)", red "<What> failed: <reason>" or orange
            "Stopped after <what was done>"; the buttons come back. One napari notification
            when the run finished or failed (none for Stop: the user is on the tab).

The job function runs on a worker thread as fn(cancel, progress) where progress(done, total,
text) may be called from any thread; progress and the end reach the UI only through Qt
signals. Results are handed to the tab's callbacks on the UI thread (layers are added there).
"""
import atexit
import os
import threading
import time
import traceback

from qtpy.QtCore import QObject, QThread, Signal

from sprout_core.boundary import Cancelled

_LIVE = set()       # running jobs: kept alive until their thread ends (no parent on purpose)


def check_cancel(cancel):
    """Raise Cancelled if the job's Stop event is set."""
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def reason(e):
    """A readable failure reason for the status line."""
    text = str(e).strip()
    if isinstance(e, MemoryError):
        return "not enough memory" + (f" ({text})" if text else "") + "; try an ROI or a smaller image"
    if not text:
        return type(e).__name__
    if type(e) in (Exception, RuntimeError, ValueError, AssertionError):
        return text
    return f"{type(e).__name__}: {text}"


class FolderWatch:
    """The files under `folder` before a run, to tell which ones the run wrote."""

    def __init__(self, folder):
        self.folder = folder
        self.before = self._scan()

    def _scan(self):
        out = {}
        if self.folder and os.path.isdir(self.folder):
            for root, _, files in os.walk(self.folder):
                for f in files:
                    p = os.path.join(root, f)
                    try:
                        out[p] = os.stat(p).st_mtime_ns
                    except OSError:
                        pass
        return out

    def written(self):
        """Number of files created or changed since the snapshot."""
        return sum(1 for p, t in self._scan().items() if self.before.get(p) != t)


def files_note(n):
    if n is None:
        return ""
    if n == 0:
        return "; no file written"
    return f"; {n} file{'' if n == 1 else 's'} already written stay{'s' if n == 1 else ''} in the output folder"


def stopped_text(done, total, unit, files=None):
    """'Stopped after 3 of 8 <unit>' (+ what was written: files = count, or None)."""
    what = f"Stopped after {done} of {total} {unit}" if total else "Stopped before the first step"
    return what + files_note(files)


class Job(QThread):
    """fn(cancel, progress) on a worker thread. Ends with one `ended(kind, payload)`:
    ("ok", result), ("stopped", exception) or ("error", exception)."""

    progressed = Signal(int, int, str)
    ended = Signal(str, object)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancel = threading.Event()
        self.started_at = None

    def run(self):
        try:
            result = self.fn(self.cancel, self._progress)
            if self.cancel.is_set():            # Stop came after the last check: still stopped
                raise Cancelled()
        except Cancelled as e:
            self.ended.emit("stopped", e)
        except Exception as e:                  # MemoryError included
            traceback.print_exception(type(e), e, e.__traceback__)   # the detail, in the console only
            self.ended.emit("error", e)
        else:
            self.ended.emit("ok", result)

    def _progress(self, done, total, text=""):
        self.progressed.emit(int(done), int(total), str(text or ""))

    def stop(self):
        self.cancel.set()


def _stop_all_at_exit():
    for job in list(_LIVE):
        job.stop()
        job.wait(10000)


atexit.register(_stop_all_at_exit)


class RunControl(QObject):
    """Runs one job at a time for a tab and drives its buttons, progress bar and status line.

    status       StatusLine of the run
    run_buttons  disabled while running
    stop_button  enabled while running (the tab connects its click to stop())
    bar          QProgressBar or None; hide_bar: shown only while running
    notify       callable(level, text), level 'info' / 'error': the napari notification
                 (passed by the tab, so its module's show_info / show_error are used)
    """

    def __init__(self, parent, status, run_buttons, stop_button=None, bar=None, hide_bar=True,
                 notify=None):
        super().__init__(parent)
        self.status = status
        self.run_buttons = list(run_buttons)
        self.stop_button = stop_button
        self.bar = bar
        self.hide_bar = hide_bar
        self.notify = notify
        self.job = None
        self.done = self.total = 0
        self.progress_text = ""
        self._spec = None
        # the tab is closed while a job runs: stop it (not a slot of self, which goes with it)
        holder = self._holder = {"job": None}
        parent.destroyed.connect(lambda *_: holder["job"] is not None and holder["job"].stop())
        if bar is not None and hide_bar:
            bar.setVisible(False)

    @property
    def running(self):
        return self.job is not None and self._spec is not None

    def start(self, fn, title, running, done=None, stopped=None, failed=None, after=None,
              level="info"):
        """Start fn(cancel, progress) unless a job is running (then returns None).

        title    "<title> failed: <reason>" ("Seed generation", "Growth", ...)
        running  the grey line while it runs ("Generating seeds…"); level='warn' for a
                 run that starts with a warning
        done(result, seconds)  on the UI thread; returns the summary str, or (summary, note),
                 or (summary, note, level); shown as "✓ <summary> (<t> s)<note>". None: the
                 callback set the status itself (and no notification).
        stopped(exc, done, total)  -> the orange text (default "Stopped after d of t steps");
                 also when Stop came after the job's last check (its result is dropped)
        failed(exc)  -> the reason (default jobs.reason)
        after(kind)  on the UI thread once everything else is done (kind: ok / stopped / error)
        """
        if self.running:
            return None
        job = Job(fn)
        self._spec = dict(title=title, running=running, done=done, stopped=stopped, failed=failed,
                          after=after, level=level, t0=time.perf_counter())
        self.job = self._holder["job"] = job
        self.done = self.total = 0
        self.progress_text = ""
        job.progressed.connect(self._on_progress)
        job.ended.connect(self._on_end)
        job.finished.connect(lambda: _LIVE.discard(job))
        _LIVE.add(job)
        self._set_running(True)
        self.status.set(running, level)
        job.start()
        return job

    def stop(self):
        """Ask the running job to stop at its next check."""
        if not self.running or self.job.cancel.is_set():
            return
        self.job.stop()
        if self.stop_button is not None:
            self.stop_button.setEnabled(False)
        self.status.set("Stopping…", "info")

    # ---------------------------------------------------------------- internals
    def _set_running(self, on):
        for b in self.run_buttons:
            b.setEnabled(not on)
        if self.stop_button is not None:
            self.stop_button.setEnabled(on)
        if self.bar is not None:
            if on:
                self.bar.setRange(0, 0)          # busy until the first real progress
                self.bar.setFormat("%v / %m")
                if self.hide_bar:
                    self.bar.setVisible(True)
            elif self.hide_bar:
                self.bar.setVisible(False)

    def _on_progress(self, done, total, text):
        if self.sender() is not self.job or self._spec is None:
            return
        self.done, self.total, self.progress_text = done, total, text
        if self.bar is not None:
            self.bar.setRange(0, max(total, 1))
            self.bar.setValue(done)
        if not self.job.cancel.is_set():
            spec = self._spec
            self.status.set(f"{spec['running']} ({text})" if text else spec["running"], spec["level"])

    def _on_end(self, kind, payload):
        if self.sender() is not self.job or self._spec is None:
            return
        spec, self._spec = self._spec, None
        secs = time.perf_counter() - spec["t0"]
        self._set_running(False)
        if kind == "ok":
            try:
                self._finished(spec, payload, secs)
            except Exception as e:                # e.g. adding the layers failed
                traceback.print_exception(type(e), e, e.__traceback__)
                kind = "error"
                self._failed(spec, e)
        elif kind == "stopped":
            text = (spec["stopped"](payload, self.done, self.total) if spec["stopped"]
                    else stopped_text(self.done, self.total, "steps"))
            self.status.set(text, "warn")
        else:
            self._failed(spec, payload)
        if spec["after"] is not None:
            spec["after"](kind)

    def _finished(self, spec, result, secs):
        out = spec["done"](result, secs) if spec["done"] else spec["title"] + " done"
        if out is None:
            return
        out = (out,) if isinstance(out, str) else tuple(out)
        summary, note, level = out[0], (out[1:2] or ("",))[0], (out[2:3] or ("ok",))[0]
        text = f"{summary} ({secs:.1f} s){note}"
        self.status.set(text, level)
        if self.notify is not None:
            self.notify("info", text)

    def _failed(self, spec, e):
        why = spec["failed"](e) if spec["failed"] else reason(e)
        text = f"{spec['title']} failed: {why}"
        self.status.set(text, "error")
        if self.notify is not None:
            self.notify("error", text)


__all__ = ["Job", "RunControl", "FolderWatch", "files_note", "stopped_text", "reason", "check_cancel",
           "Cancelled"]
