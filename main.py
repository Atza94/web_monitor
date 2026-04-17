"""
Spectrum Web – Interfaccia web per due Rigol DSA1030.

Avvio:
    python main.py

L'app sarà disponibile su http://localhost:8080
"""

from __future__ import annotations
import asyncio
import errno
import io
import logging
import os
from collections import deque
from datetime import datetime, timezone
import numpy as np
from nicegui import app, run, ui

from config import (
    DEFAULT_CENTER_FREQ_HZ,
    DEFAULT_RBW_HZ,
    DEFAULT_SPAN_HZ,
    DEFAULT_VBW_HZ,
    FREQ_MAX_HZ,
    FREQ_MIN_HZ,
    MEDICINA_LONGITUDE_DEG,
    OPERATOR_PASSWORD,
    RBW_OPTIONS_HZ,
    STORAGE_SECRET,
    TRACE_UPDATE_INTERVAL_S,
    VBW_OPTIONS_HZ,
    SCPI_COMMANDS,
    SERVPORT,
    SERV_HOST,
    SUB_AXES,
    RECORDINGS_DIR,
)
from instruments import InstrumentManager, InstrumentParams, TraceData, ServManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("uvicorn").addFilter(
    lambda record: "" not in record.getMessage()
)

# ── Manager globale (singleton) ───────────────────────────
mgr = InstrumentManager()
app.on_shutdown(mgr.disconnect_all)

# ── Stato globale condiviso tra tutti i client ─────────────
_latest_traces:     dict[str, TraceData | None]      = {k: None  for k in mgr.instrument_ids}
_trace_versions:    dict[str, int]                   = {k: 0     for k in mgr.instrument_ids}
_latest_traces2:    dict[str, TraceData | None]      = {k: None  for k in mgr.instrument_ids}
_trace2_versions:   dict[str, int]                   = {k: 0     for k in mgr.instrument_ids}
_reading:           dict[str, bool]                  = {k: False for k in mgr.instrument_ids}
_continuous_active: dict[str, bool]                  = {k: False for k in mgr.instrument_ids}
_bg_tasks:          dict[str, asyncio.Task | None]   = {k: None  for k in mgr.instrument_ids}


HISTORY_SIZE = 8             # sweep conservati in buffer FIFO                                      
_AZ_EL_HISTORY = 180          # punti massimi conservati nella history del grafico antenna
_TRACK_THRESHOLD_DEG = 1.00  # scarto minimo per registrare un nuovo punto


_ON_SOURCE_MAP: dict[int, tuple[str, str]] = {
    0: ("OFF SOURCE", "text-red-500"),
    1: ("ON SOURCE",  "text-green-500"),
    2: ("OFFSET",     "text-yellow-500"),
}


def is_authenticated() -> bool:
    return app.storage.user.get("authenticated", False)


async def _background_reader(instr_id: str) -> None:
    """Task asyncio (uno per strumento) che legge le tracce e aggiorna la cache globale."""
    while _continuous_active[instr_id]:
        if not _reading[instr_id] and mgr.is_connected(instr_id):
            _reading[instr_id] = True
            try:
                trace = await run.io_bound(mgr.read_trace, instr_id, 1)
                _latest_traces[instr_id] = trace
                _trace_versions[instr_id] += 1
                trace2 = await run.io_bound(mgr.read_trace, instr_id, 2)
                _latest_traces2[instr_id] = trace2
                _trace2_versions[instr_id] += 1
            except Exception as exc:
                if "timeout" in str(exc).lower() or "VI_ERROR_TMO" in str(exc):
                    logger.warning("Timeout VISA su %s — strumento non risponde", instr_id)
                else:
                    logger.warning("Errore lettura traccia %s: %s", instr_id, exc)
            finally:
                _reading[instr_id] = False
        await asyncio.sleep(TRACE_UPDATE_INTERVAL_S)


# ── Helper di formattazione ───────────────────────────────

def fmt_freq(hz: float) -> str:
    """Formatta una frequenza in modo leggibile."""
    if hz >= 1e9:
        return f"{hz / 1e9:.4f} GHz"
    if hz >= 1e6:
        return f"{hz / 1e6:.3f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.1f} kHz"
    return f"{hz:.0f} Hz"


def freq_options_map(options_hz: list[float]) -> dict[float, str]:
    """Crea un dict valore→label per i select di RBW/VBW."""
    return {v: fmt_freq(v) for v in options_hz}


# ── Calcolo tempo siderale ────────────────────────────────

def _calc_lst(utc: datetime) -> float:
    """Local Sidereal Time in hours per Medicina (IAU GMST formula)."""
    jd = utc.timestamp() / 86400.0 + 2440587.5   # Julian Date
    d = jd - 2451545.0                            # giorni da J2000.0
    gmst_deg = (280.46061837 + 360.98564736629 * d) % 360.0
    return ((gmst_deg + MEDICINA_LONGITUDE_DEG) % 360.0) / 15.0  # ore


def _fmt_hms(hours: float) -> str:
    h = int(hours)
    m = int((hours - h) * 60)
    s = int(((hours - h) * 60 - m) * 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Pannello per singolo strumento ────────────────────────

class AnalyzerPanel:
    """
    Widget NiceGUI per un singolo analizzatore di spettro.
    Gestisce connessione, grafico live, e controlli parametri.
    """

    def __init__(self, instr_id: str) -> None:
        self.instr_id = instr_id
        self.label = mgr.get_label(instr_id)
        self.chart = None
        self.status_icon = None
        self.status_label = None
        self.connect_btn = None
        self.params_container = None
        self.continuous_btn = None
        self.sweep_status = None
        self._displayed_version: int = -1
        self._displayed_version2: int = -1
        self._show_trace2: bool = False
        self.trace2_btn = None
        self._display_timer = None
        self._sweep_info_label = None
        self._history_btn = None
        self._show_history: bool = False
        self._history: deque = deque(maxlen=HISTORY_SIZE)
        self._sweep_time_s: float = 0.0
        self._recording: bool = False
        self._record_file: io.TextIOWrapper | None = None
        self._rec_btn = None
        self._build_ui()

    def _build_ui(self) -> None:
        with ui.card().classes("w-full"):
            # ── Header con stato connessione ──────────────
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(self.label).classes("text-xl font-bold")
                with ui.row().classes("items-center gap-2"):
                    self.status_icon = ui.icon("circle").classes("text-red-500")
                    self.status_label = ui.label("Disconnesso").classes("text-sm")
                    self.connect_btn = ui.button(
                        "Connetti",
                        on_click=self._toggle_connection,
                    ).props("dense")

            ui.separator()

            # ── Grafico spettro (Plotly) ──────────────────
            self.chart = ui.plotly(
                {
                    "data": [
                        {
                            "x": [],
                            "y": [],
                            "type": "scatter",
                            "mode": "lines",
                            "name": "Trace 1 (CW)",
                            "line": {"color": "#2196F3", "width": 1.5},
                        },
                        {
                            "x": [],
                            "y": [],
                            "type": "scatter",
                            "mode": "lines",
                            "name": "Trace 2 (MaxHold)",
                            "line": {"color": "#FF9800", "width": 1.5},
                            "visible": "legendonly",
                        },
                        *[
                            {
                                "x": [],
                                "y": [],
                                "type": "scatter",
                                "mode": "lines",
                                "name": "history",
                                "line": {"color": "#444444", "width": 0.8},
                                "visible": False,
                                "showlegend": False,
                                "hoverinfo": "skip",
                            }
                            for _ in range(HISTORY_SIZE)
                        ],
                    ],
                    "layout": {
                        "title": None,
                        "xaxis": {"title": "Frequenza (MHz)", "tickformat": ".2f"},
                        "yaxis": {"title": "Ampiezza (dBm)", "range": [-120, 0]},
                        "margin": {"l": 60, "r": 20, "t": 10, "b": 50},
                        "height": 350,
                        "uirevision": "constant",  # evita reset zoom durante update
                        "legend": {"x": 0.01, "y": 0.99},
                    },
                }
            ).classes("w-full")

            with ui.row().classes("items-center gap-2 -mt-2 mb-1"):
                self.trace2_btn = ui.button(
                    "MaxHold", on_click=self._toggle_trace2
                ).props("dense outline size=sm")
                self._history_btn = ui.button(
                    "History", on_click=self._toggle_history
                ).props("dense outline size=sm")
                ui.button(
                    icon="download", on_click=self._download_csv
                ).props("dense outline size=sm").tooltip("Scarica CSV")
                self._sweep_info_label = ui.label("").classes("text-xs text-gray-500 font-mono ml-2")

            # ── Controlli parametri ───────────────────────
            self.params_container = ui.column().classes("w-full")
            with self.params_container:
                self._build_controls()

            # Disabilita controlli finché non connesso
            self.params_container.set_visibility(False)

    def _build_controls(self) -> None:
        """Crea i controlli per center freq, span, RBW, VBW."""
        with ui.row().classes("w-full flex-wrap gap-4 items-end"):
            # Center Frequency
            with ui.column().classes("gap-1"):
                ui.label("Center Freq (MHz)").classes("text-xs text-gray-500")
                self.center_input = ui.number(
                    value=DEFAULT_CENTER_FREQ_HZ / 1e6,
                    format="%.3f",
                    min=FREQ_MIN_HZ / 1e6,
                    max=FREQ_MAX_HZ / 1e6,
                    step=1.0,
                ).classes("w-36")

            # Span
            with ui.column().classes("gap-1"):
                ui.label("Span (MHz)").classes("text-xs text-gray-500")
                self.span_input = ui.number(
                    value=DEFAULT_SPAN_HZ / 1e6,
                    format="%.3f",
                    min=0,
                    max=FREQ_MAX_HZ / 1e6,
                    step=1.0,
                ).classes("w-36")

            # RBW
            with ui.column().classes("gap-1"):
                ui.label("RBW").classes("text-xs text-gray-500")
                self.rbw_select = ui.select(
                    options=freq_options_map(RBW_OPTIONS_HZ),
                    value=DEFAULT_RBW_HZ,
                ).classes("w-36")

            # VBW
            with ui.column().classes("gap-1"):
                ui.label("VBW").classes("text-xs text-gray-500")
                self.vbw_select = ui.select(
                    options=freq_options_map(VBW_OPTIONS_HZ),
                    value=DEFAULT_VBW_HZ,
                ).classes("w-36")

            # Bottone applica
            ui.button("Applica", on_click=self._apply_params).props(
                "color=primary dense"
            )

            # Bottone leggi dallo strumento
            ui.button("Leggi strumento", on_click=self._read_params).props(
                "color=secondary dense flat"
            )

        # ── Controlli sweep ───────────────────────────────
        with ui.row().classes("w-full items-center gap-4 mt-2"):
            ui.button("Single Sweep", on_click=self._single_sweep).props(
                "dense outline"
            )
            self.continuous_btn = ui.button(
                "Continuous Sweep", on_click=self._toggle_continuous
            ).props("dense outline")
            self.sweep_status = ui.label("").classes("text-xs text-gray-400")
            self._rec_btn = (
                ui.button(icon="fiber_manual_record", on_click=self._toggle_recording)
                .props("dense outline size=sm")
                .tooltip("Avvia / Ferma registrazione")
            )
            self._rec_btn.set_enabled(False)

        # Campo comando libero
            with ui.row().classes("w-full items-center gap-4 mt-2"):
                ui.label("SCPI").classes("text-xs text-violet-500")
                self.command = ui.input(
                    label="Comando SCPI",
                    placeholder="e.g. :SENSe:FREQuency:CENTer?",
                    autocomplete=list(SCPI_COMMANDS.keys()),
                ).classes("w-64") 
                (ui.button(color="red", on_click=lambda: self.command.set_value(""), icon="delete")
                    .props("dense outline"))
                ui.button("Invia", on_click=self._send_scpi).props(
                    "color=green dense"
                )
                self.scpi_response = ui.label("").classes("text-xs text-green-300 font-mono")
                 
                                     

    @property
    def last_trace(self) -> TraceData | None:
        """Ultima traccia disponibile dalla cache globale."""
        return _latest_traces.get(self.instr_id)

    # ── Aggiornamento display (legge da cache, nessun I/O) ─

    async def _refresh_chart(self) -> None:
        needs_update = False

        version = _trace_versions.get(self.instr_id, 0)
        if version != self._displayed_version:
            trace = _latest_traces.get(self.instr_id)
            if trace is not None:
                # Prima di sovrascrivere, salva il dato corrente in history
                if self._show_history and self.chart.figure["data"][0]["x"]:
                    self._history.appendleft({
                        "x": list(self.chart.figure["data"][0]["x"]),
                        "y": list(self.chart.figure["data"][0]["y"]),
                    })
                    self._update_ghost_traces()

                self._displayed_version = version
                self.chart.figure["data"][0]["x"] = (trace.frequencies / 1e6).tolist()
                self.chart.figure["data"][0]["y"] = trace.amplitudes.tolist()
                self._update_sweep_info(num_points=trace.num_points)
                needs_update = True

        version2 = _trace2_versions.get(self.instr_id, 0)
        if version2 != self._displayed_version2:
            trace2 = _latest_traces2.get(self.instr_id)
            if trace2 is not None:
                self._displayed_version2 = version2
                self.chart.figure["data"][1]["x"] = (trace2.frequencies / 1e6).tolist()
                self.chart.figure["data"][1]["y"] = trace2.amplitudes.tolist()
                needs_update = True
                if self._recording:
                    self._write_record_row(trace2)

        if needs_update:
            self.chart.update()

    # ── Connessione ───────────────────────────────────────

    async def _toggle_connection(self) -> None:
        if mgr.is_connected(self.instr_id):
            await self._disconnect()
        else:
            await self._connect()

    async def _connect(self) -> None:
        self.connect_btn.props("loading")
        try:
            await run.io_bound(mgr.connect, self.instr_id)
            self._set_connected(True)
            await self._read_params()
            await self._update_trace()
            ui.notify(f"{self.label} connesso", type="positive")
        except ConnectionError as exc:
            ui.notify(str(exc), type="negative")
        finally:
            self.connect_btn.props(remove="loading")

    async def _disconnect(self) -> None:
        self._stop_continuous()
        mgr.disconnect(self.instr_id)
        self._set_connected(False)
        ui.notify(f"{self.label} disconnesso", type="warning")

    def _set_connected(self, connected: bool) -> None:
        if connected:
            self.status_icon.classes(replace="text-green-500")
            self.status_label.text = "Connesso"
            self.connect_btn.text = "Disconnetti"
            # Avvia il display timer per questo client
            self._displayed_version = -1
            self._displayed_version2 = -1
            self._display_timer = ui.timer(TRACE_UPDATE_INTERVAL_S, self._refresh_chart)
            # Rifletti lo stato globale del continuous (altri client potrebbero averlo attivato)
            if _continuous_active[self.instr_id]:
                self.continuous_btn.props("color=negative")
                self.continuous_btn.text = "Stop"
                self.sweep_status.text = "● Continuous"
                self.sweep_status.classes(replace="text-xs text-green-400")
        else:
            self.status_icon.classes(replace="text-red-500")
            self.status_label.text = "Disconnesso"
            self.connect_btn.text = "Connetti"
            # Ferma il display timer di questo client
            if self._display_timer:
                self._display_timer.cancel()
                self._display_timer = None
        self._update_controls_visibility()

    def _update_controls_visibility(self) -> None:
        """Aggiorna visibilità dei controlli in base a connessione e autenticazione."""
        auth = is_authenticated()
        self.connect_btn.set_visibility(auth)
        self.params_container.set_visibility(auth and mgr.is_connected(self.instr_id))

    def apply_auth(self, authenticated: bool) -> None:
        """Chiamato dalla pagina quando lo stato di autenticazione cambia."""
        self._update_controls_visibility()

    # ── Sweep ──────────────────────────────────────────────

    async def _single_sweep(self) -> None:
        """Esegue una singola lettura traccia."""
        await self._update_trace()

    async def _toggle_continuous(self) -> None:
        """Attiva/disattiva il background reader globale per questo strumento."""
        if _continuous_active[self.instr_id]:
            self._stop_continuous()
        else:
            self._start_continuous()

    def _start_continuous(self) -> None:
        _continuous_active[self.instr_id] = True
        if _bg_tasks[self.instr_id] is None or _bg_tasks[self.instr_id].done():
            _bg_tasks[self.instr_id] = asyncio.ensure_future(
                _background_reader(self.instr_id)
            )
        self.continuous_btn.props("color=negative")
        self.continuous_btn.text = "Stop"
        self.sweep_status.text = "● Continuous"
        self.sweep_status.classes(replace="text-xs text-green-400")
        self._rec_btn.set_enabled(True)

    def _stop_continuous(self) -> None:
        _continuous_active[self.instr_id] = False  # il task esce dal loop al prossimo ciclo
        self.continuous_btn.props(remove="color")
        self.continuous_btn.text = "Continuous Sweep"
        self.sweep_status.text = ""
        self.sweep_status.classes(replace="text-xs text-gray-400")
        if self._recording:
            self._stop_recording()
        self._rec_btn.set_enabled(False)

    # ── Single sweep (aggiorna cache globale) ─────────────

    async def _update_trace(self) -> None:
        """Legge TRACE1 e TRACE2 una volta e aggiorna la cache globale."""
        if not mgr.is_connected(self.instr_id) or _reading[self.instr_id]:
            return
        _reading[self.instr_id] = True
        try:
            trace: TraceData = await run.io_bound(mgr.read_trace, self.instr_id, 1)
            _latest_traces[self.instr_id] = trace
            _trace_versions[self.instr_id] += 1
            trace2: TraceData = await run.io_bound(mgr.read_trace, self.instr_id, 2)
            _latest_traces2[self.instr_id] = trace2
            _trace2_versions[self.instr_id] += 1
        except Exception as exc:
            if "timeout" in str(exc).lower() or "VI_ERROR_TMO" in str(exc):
                ui.notify(f"{self.label}: timeout — strumento non risponde", type="negative")
            logger.warning("Errore lettura traccia %s: %s", self.instr_id, exc)
        finally:
            _reading[self.instr_id] = False

    def _toggle_trace2(self) -> None:
        """Mostra/nasconde Trace 2 (MaxHold) sul grafico."""
        self._show_trace2 = not self._show_trace2
        self.chart.figure["data"][1]["visible"] = True if self._show_trace2 else "legendonly"
        self.chart.update()
        if self._show_trace2:
            self.trace2_btn.props("color=orange")
        else:
            self.trace2_btn.props(remove="color")

    def _toggle_history(self) -> None:
        """Attiva/disattiva la visualizzazione degli ultimi N sweep sul grafico."""
        self._show_history = not self._show_history
        if self._show_history:
            self._history_btn.props("color=purple")
        else:
            self._history.clear()
            self._update_ghost_traces()
            self.chart.update()
            self._history_btn.props(remove="color")

    def _update_ghost_traces(self) -> None:
        """Aggiorna i ghost traces con i dati della history."""
        for i, hist in enumerate(self._history):
            self.chart.figure["data"][2 + i]["x"] = hist["x"]
            self.chart.figure["data"][2 + i]["y"] = hist["y"]
            self.chart.figure["data"][2 + i]["visible"] = True
        for i in range(len(self._history), HISTORY_SIZE):
            self.chart.figure["data"][2 + i]["x"] = []
            self.chart.figure["data"][2 + i]["y"] = []
            self.chart.figure["data"][2 + i]["visible"] = False

    def _update_sweep_info(self, sweep_time: float | None = None, num_points: int | None = None) -> None:
        """Aggiorna il label con num_points e sweep time."""
        parts = []
        if num_points is not None:
            parts.append(f"{num_points} pts")
        t = sweep_time if sweep_time is not None else self._sweep_time_s
        if t > 0:
            parts.append(f"{t:.2f} s")
        self._sweep_info_label.set_text("  ".join(parts))

    def _download_csv(self) -> None:
        """Scarica le tracce correnti come CSV con header UTC/LST."""
        trace = _latest_traces.get(self.instr_id)
        if trace is None:
            ui.notify("Nessuna traccia disponibile", type="warning")
            return
        trace2 = _latest_traces2.get(self.instr_id)
        now = datetime.now(timezone.utc)
        lst = _calc_lst(now)
        s = ServManager.state
        buf = io.StringIO()
        buf.write(f"# Spettro – {self.label}\n")
        buf.write(f"# UTC: {now.strftime('%Y-%m-%d %H:%M:%SZ')}\n")
        buf.write(f"# LST: {_fmt_hms(lst)}\n")
        buf.write(f"# Az: {s.az_deg:.3f}°  El: {s.el_deg:.3f}°\n")
        buf.write(f"# Points: {trace.num_points}  Sweep: {self._sweep_time_s:.2f} s\n")
        buf.write("freq_mhz,trace1_dbm")
        if trace2 is not None:
            buf.write(",trace2_dbm")
        buf.write("\n")
        for i, (f, a) in enumerate(zip(trace.frequencies / 1e6, trace.amplitudes)):
            row = f"{f:.6f},{a:.4f}"
            if trace2 is not None and i < len(trace2.amplitudes):
                row += f",{trace2.amplitudes[i]:.4f}"
            buf.write(row + "\n")

        filename = f"spectrum_{self.instr_id}_{now.strftime('%Y%m%dT%H%M%SZ')}.csv"
        ui.download(buf.getvalue().encode(), filename)

    # ── Registrazione stream ───────────────────────────────

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        now = datetime.now(timezone.utc)
        try:
            os.makedirs(RECORDINGS_DIR, exist_ok=True)
            path = os.path.join(
                RECORDINGS_DIR,
                f"{self.instr_id}_{now.strftime('%Y%m%dT%H%M%SZ')}.csv",
            )
            self._record_file = open(path, "w", buffering=1)  # line-buffered
        except OSError as exc:
            ui.notify(f"Impossibile aprire il file: {exc}", type="negative")
            return

        trace = _latest_traces.get(self.instr_id)
        s = ServManager.state
        lst = _calc_lst(now)
        n = trace.num_points if trace else 601
        t1_cols = ",".join(f"t1_{i}" for i in range(n))
        t2_cols = ",".join(f"t2_{i}" for i in range(n))

        self._record_file.write(f"# Spettro – {self.label}\n")
        self._record_file.write(f"# Start UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._record_file.write(f"# LST: {_fmt_hms(lst)}\n")
        self._record_file.write(f"# Az: {s.az_deg:.3f}°  El: {s.el_deg:.3f}°\n")
        if trace is not None:
            self._record_file.write(
                f"# Freq: {trace.start_freq/1e6:.3f}–{trace.stop_freq/1e6:.3f} MHz"
                f"  Points: {trace.num_points}\n"
            )
        self._record_file.write(f"utc,az_deg,el_deg,{t1_cols},{t2_cols}\n")

        self._recording = True
        self._rec_btn.props("color=negative")
        ui.notify("Registrazione avviata", type="positive")

    def _stop_recording(self, notify: bool = True) -> None:
        self._recording = False
        if self._record_file:
            try:
                self._record_file.close()
            except OSError:
                pass
            self._record_file = None
        self._rec_btn.props(remove="color")
        if notify:
            ui.notify("Registrazione terminata", type="info")

    def _write_record_row(self, trace2: TraceData) -> None:
        """Scrive una riga nel file di registrazione (chiamato ad ogni nuovo sweep)."""
        trace1 = _latest_traces.get(self.instr_id)
        if trace1 is None or self._record_file is None:
            return
        try:
            now = datetime.now(timezone.utc)
            s = ServManager.state
            row = f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            row += f",{s.az_deg:.3f},{s.el_deg:.3f}"
            row += "," + ",".join(f"{v:.4f}" for v in trace1.amplitudes)
            row += "," + ",".join(f"{v:.4f}" for v in trace2.amplitudes)
            self._record_file.write(row + "\n")
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                ui.notify("Disco pieno — registrazione interrotta", type="negative")
            else:
                ui.notify(f"Errore scrittura: {exc}", type="negative")
            self._stop_recording(notify=False)

    # ── Parametri ─────────────────────────────────────────

    async def _apply_params(self) -> None:
        """Invia i parametri dai controlli allo strumento."""
        if not mgr.is_connected(self.instr_id):
            ui.notify("Strumento non connesso", type="negative")
            return
        try:
            await run.io_bound(mgr.set_center_freq, self.instr_id, self.center_input.value * 1e6)
            await run.io_bound(mgr.set_span, self.instr_id, self.span_input.value * 1e6)
            await run.io_bound(mgr.set_rbw, self.instr_id, self.rbw_select.value)
            await run.io_bound(mgr.set_vbw, self.instr_id, self.vbw_select.value)
            ui.notify("Parametri applicati", type="positive")
        except Exception as exc:
            ui.notify(f"Errore: {exc}", type="negative")

    async def _read_params(self) -> None:
        """Legge i parametri correnti dallo strumento e aggiorna i controlli."""
        if not mgr.is_connected(self.instr_id):
            return
        try:
            p: InstrumentParams = await run.io_bound(mgr.get_params, self.instr_id)
            self.center_input.value = p.center_freq_hz / 1e6
            self.span_input.value = p.span_hz / 1e6
            self.rbw_select.value = p.rbw_hz
            self.vbw_select.value = p.vbw_hz
            self._sweep_time_s = p.sweep_time_s
            self._update_sweep_info(sweep_time=p.sweep_time_s)
        except Exception as exc:
            ui.notify(f"Errore lettura parametri: {exc}", type="negative")

    # ── Handler comando SCPI libero ────────────────────────────────
    # Permette di inviare comandi SCPI arbitrari dallo stesso pannello dei parametri
    async def _send_scpi(self) -> None:
        """Invia il comando SCPI libero e mostra l'eventuale risposta."""
        if not mgr.is_connected(self.instr_id):
            ui.notify("Strumento non connesso", type="negative")
            return
        cmd = self.command.value.strip()
        if not cmd:
            return
        try:
            result = await run.io_bound(mgr.free_field, self.instr_id, cmd)
            if result is not None:
                self.scpi_response.text = f"→ {result}"
            else:
                self.scpi_response.text = f"→ OK"
                ui.notify(f"Comando inviato", type="positive")
        except Exception as exc:
            self.scpi_response.text = ""
            ui.notify(f"Errore SCPI: {exc}", type="negative")

# ── Antenna Panel ─────────────────────────────────────────


class AntennaPanel:
    def __init__(self):
        self._track_az: deque[float] = deque(maxlen=_AZ_EL_HISTORY)
        self._track_el: deque[float] = deque(maxlen=_AZ_EL_HISTORY)

        with ui.card().classes("w-full"):
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("Antenna").classes("text-lg font-bold")
                self._status_dot = ui.icon("circle", size="sm").classes("text-red-500")

            with ui.grid(columns=4).classes("w-full gap-2"):
                # Az/El
                with ui.card().classes("col-span-1"):
                    ui.label("Posizione").classes("font-bold text-sm opacity-70")
                    with ui.grid(columns=2):
                        ui.label("Az att.")
                        self._az = ui.label("—").classes("font-mono")
                        ui.label("El att.")
                        self._el = ui.label("—").classes("font-mono")
                        ui.label("Az cmd.")
                        self._az_cmd = ui.label("—").classes("font-mono opacity-60")
                        ui.label("El cmd.")
                        self._el_cmd = ui.label("—").classes("font-mono opacity-60")
                        ui.label("Err. punt.")
                        self._pterr = ui.label("—").classes("font-mono")

                # On source
                with ui.card().classes("col-span-1"):
                    ui.label("Stato").classes("font-bold text-sm opacity-70")
                    self._onsrc = ui.label("—").classes("text-2xl font-bold text-center w-full")
                    with ui.grid(columns=2):
                        ui.label("Sorgente").classes("font-bold text-sm opacity-70")
                        self._nmsrc = ui.label("—").classes("font-bold text-sm text-end w-full")
                        ui.label("Temp.")
                        self._temp = ui.label("—").classes("font-mono")
                        ui.label("Hum.")
                        self._hum = ui.label("—").classes("font-mono")
                        ui.label("Press.")
                        self._pres = ui.label("—").classes("font-mono")
                        ui.label("Wind")
                        self._wind = ui.label("—").classes("font-mono")
                # Receiver
                with ui.card().classes("col-span-1"):
                    ui.label("Ricevitore").classes("font-bold text-sm opacity-70")
                    with ui.grid(columns=2):
                        ui.label("Code")
                        self._rx = ui.label("—").classes("font-mono")
                        ui.label("LO")
                        self._lo = ui.label("—").classes("font-mono")
                        ui.label("Noise cal")
                        self._cal = ui.label("—").classes("font-mono")

                # Subriflettore
                with ui.card().classes("col-span-1"):
                    ui.label("Subriflettore").classes("font-bold text-sm opacity-70")
                    with ui.grid(columns=3):
                        ui.label("Asse").classes("text-xs opacity-60")
                        ui.label("Cmd").classes("text-xs opacity-60")
                        ui.label("Act").classes("text-xs opacity-60")
                        self._sub_cmd = []
                        self._sub_act = []
                        for i in SUB_AXES:
                            ui.label(f"{i}").classes("text-xs")
                            self._sub_cmd.append(ui.label("—").classes("font-mono text-xs"))
                            self._sub_act.append(ui.label("—").classes("font-mono text-xs"))

            with ui.expansion("Posizione Antenna", icon="track_changes").classes("w-full mt-2"):
                self._track_chart = ui.plotly({
                    "data": [
                        {
                            "r": [], "theta": [],
                            "type": "scatterpolar", "mode": "lines+markers",
                            "name": "Track",
                            "line": {"color": "#2196F3", "width": 1.5},
                            "marker": {"size": [], "color": [], "line": {"color": "white", "width": []}},
                            "showlegend": False,
                        },
                    ],
                    "layout": {
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "polar": {
                            "bgcolor": "rgba(40,40,40,0.5)",
                            "angularaxis": {
                                "direction": "clockwise",
                                "rotation": 90,  # Nord in cima
                                "tickmode": "array",
                                "tickvals": [0, 45, 90, 135, 180, 225, 270, 315],
                                "ticktext": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                                "tickfont": {"color": "#cccccc"},
                                "linecolor": "#555555",
                                "gridcolor": "#444444",
                            },
                            "radialaxis": {
                                "range": [90, 0],  # centro=90°El(zenith), bordo=0°El(orizzonte)
                                "tickmode": "array",
                                "tickvals": [90, 80, 70, 60, 50, 40, 30, 20, 10, 0],
                                "ticktext": ["90°", "80°", "70°", "60°", "50°", "40°", "30°", "20°", "10°", "0°"],
                                "tickfont": {"size": 9, "color": "#aaaaaa"},
                                "linecolor": "#555555",
                                "gridcolor": "#444444",
                            },
                        },
                        "margin": {"l": 40, "r": 40, "t": 40, "b": 40},
                        "height": 500,
                        "uirevision": "constant",
                        "font": {"color": "#cccccc"},
                    },
                }).classes("w-full")

        ui.timer(2.0, self._refresh)

    def _refresh(self):
        s = ServManager.state

        # dot rosso/verde in base alla connessione
        connected = ServManager.connected
        self._status_dot.classes(
            remove="text-green-500 text-red-500",
            add="text-green-500" if connected else "text-red-500"
        )

        self._az.set_text(f"{s.az_deg:.3f}°")
        self._el.set_text(f"{s.el_deg:.3f}°")
        self._az_cmd.set_text(f"{s.az_cmd:.3f}°")
        self._el_cmd.set_text(f"{s.el_cmd:.3f}°")
        self._pterr.set_text(f"{s.point_err:.4f}°")

        _CAL_COLOR_MAP: dict[int,str] = {
            3: "text-blue-400",
            4: "text-cyan-400",
        }

        label, color = _ON_SOURCE_MAP.get(s.on_source, ("?", "text-gray-400"))
        if s.on_source == 1 and s.noise_cal in _CAL_COLOR_MAP:
            color = _CAL_COLOR_MAP[s.noise_cal]
            
        self._onsrc.set_text(label)
        self._onsrc.classes(
            remove="text-red-500 text-green-500 text-yellow-500 text-blue-400 text-gray-400",
            add=color
        )

        self._nmsrc.set_text(s.name_src)
        self._rx.set_text(s.rx_type)
        self._lo.set_text(f"{s.lo_mhz:.1f} MHz")
        self._cal.set_text("ON" if s.noise_cal else "off")
        self._temp.set_text(f"{s.temp:.0f} °C")
        self._hum.set_text(f"{s.hum:.0f} %")
        self._pres.set_text(f"{s.pres:.2f} Pa")
        self._wind.set_text(f"{s.wind:.1f} Km/h")

        for i in range(5):
            self._sub_cmd[i].set_text(f"{s.sub_cmd[i]:.2f}")
            self._sub_act[i].set_text(f"{s.sub_act[i]:.2f}")

        if not self._track_az or (
            (s.az_deg - self._track_az[-1]) ** 2 +
            (s.el_deg - self._track_el[-1]) ** 2
        ) ** 0.5 >= _TRACK_THRESHOLD_DEG:
            self._track_az.append(s.az_deg)
            self._track_el.append(s.el_deg)
        az_list = list(self._track_az)
        el_list = list(self._track_el)
        n = len(az_list)
        sizes  = [3]         * (n - 1) + [9]
        colors = ["#2196F3"] * (n - 1) + ["#FF4444"]
        widths = [0]         * (n - 1) + [2]
        self._track_chart.figure["data"] = [{
            "r": el_list,
            "theta": az_list,
            "type": "scatterpolar",
            "mode": "lines+markers",
            "name": "Track",
            "line": {"color": "#2196F3", "width": 1.5},
            "marker": {"size": sizes, "color": colors, "line": {"color": "white", "width": widths}},
            "showlegend": False,
        }]
        self._track_chart.update()

# ── Pannello di confronto polarizzazioni ──────────────────
class ComparisonPanel:
    """Overlay delle tracce di due analizzatori per confronto polarizzazioni."""

    def __init__(self, panels: list[AnalyzerPanel]) -> None:
        self.panels = panels
        self.timer = None

        with ui.card().classes("w-full"):
            with ui.dialog() as dialog, ui.card():
                ui.label("Connettere entrambi gli strumenti in modalità ""continuous"" per abilitare il confronto").classes("p-4")
                ui.button("Ok!", on_click=dialog.close).props("color=primary")
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Confronto Polarizzazioni").classes("text-xl font-bold")
                with ui.row().classes("items-center gap-2"):
                    ui.button("Single", on_click=self._single_update)
                    self.cont_btn = ui.button(
                        "Continuous", on_click=lambda: [self._toggle_continuous(), dialog.open()]
                    ).props("dense outline")

            ui.separator()

            colors = ["#2196F3", "#FF9800"]
            self.chart = ui.plotly(
                {
                    "data": [
                        {
                            "x": [],
                            "y": [],
                            "type": "scatter",
                            "mode": "lines",
                            "name": p.label,
                            "line": {"color": colors[i], "width": 1.5},
                        }
                        for i, p in enumerate(panels)
                    ],
                    "layout": {
                        "xaxis": {"title": "Frequenza (MHz)", "tickformat": ".2f"},
                        "yaxis": {"title": "Ampiezza (dBm)", "range": [-120, 0]},
                        "margin": {"l": 60, "r": 20, "t": 10, "b": 50},
                        "height": 400,
                        "uirevision": "constant",
                        "legend": {"x": 0.01, "y": 0.99},
                    },
                }
            ).classes("w-full")

            # Delta stats
            self.delta_label = ui.label("").classes("text-xs text-gray-400 font-mono")

    async def _single_update(self) -> None:
        for i, panel in enumerate(self.panels):
            trace = panel.last_trace
            if trace is None:
                continue
            freq_mhz = (trace.frequencies / 1e6).tolist()
            amps = trace.amplitudes.tolist()
            self.chart.figure["data"][i]["x"] = freq_mhz
            self.chart.figure["data"][i]["y"] = amps

        self.chart.update()
        self._update_delta()

    def _update_delta(self) -> None:
        t0 = self.panels[0].last_trace
        t1 = self.panels[1].last_trace
        if t0 is None or t1 is None:
            return
        # Calcola delta solo se stessa banda
        if len(t0.amplitudes) == len(t1.amplitudes):
            delta = t0.amplitudes - t1.amplitudes
            self.delta_label.text = (
                f"mean: {np.mean(delta):+.2f} dBm  |  "
                f"max: {np.max(delta):+.2f} dBm  |  "
                f"min: {np.min(delta):+.2f} dBm"
            )

    def _toggle_continuous(self) -> None:
        if self.timer:
            self.timer.cancel()
            self.timer = None
            self.cont_btn.props(remove="color")
            self.cont_btn.text = "Continuous"
        else:
            self.timer = ui.timer(TRACE_UPDATE_INTERVAL_S, self._single_update)
            self.cont_btn.props("color=negative")
            self.cont_btn.text = "Stop"

# ── Pagina principale ─────────────────────────────────────

@ui.page("/")
def index():
    ui.dark_mode(True)

    panels: list[AnalyzerPanel] = []

    # ── Dialog autenticazione ─────────────────────────────
    with ui.dialog() as auth_dialog, ui.card().classes("w-80"):
        ui.label("Accesso operatore").classes("text-lg font-bold")
        pwd_input = ui.input(
            label="Password",
            password=True,
            password_toggle_button=True,
        ).classes("w-full")

        def _confirm_auth() -> None:
            if pwd_input.value == OPERATOR_PASSWORD:
                app.storage.user["authenticated"] = True
                for p in panels:
                    p.apply_auth(True)
                lock_btn.props("icon=lock_open color=positive")
                auth_dialog.close()
                ui.notify("Accesso operatore attivato", type="positive")
            else:
                ui.notify("Password errata", type="negative")
                pwd_input.value = ""

        pwd_input.on("keydown.enter", _confirm_auth)
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Annulla", on_click=auth_dialog.close).props("flat dense")
            ui.button("Accedi", on_click=_confirm_auth).props("color=primary dense")

    def _toggle_auth() -> None:
        if is_authenticated():
            app.storage.user["authenticated"] = False
            for p in panels:
                p.apply_auth(False)
            lock_btn.props("icon=lock")
            lock_btn.props(remove="color")
            ui.notify("Modalità sola lettura", type="warning")
        else:
            pwd_input.value = ""
            auth_dialog.open()

    # ── Header ────────────────────────────────────────────
    with ui.header().classes("items-center justify-between"):
        with ui.column().classes("gap-0"):
            ui.label("Spectrum Web").classes("text-2xl font-bold")
            ui.label("Rigol DSA1030 – Parabola").classes("text-sm opacity-70")
        with ui.column().classes("gap-0 items-center"):
            ut_label  = ui.label("UT  --:--:--").classes("text-sm font-mono")
            lst_label = ui.label("LST --:--:--").classes("text-sm font-mono opacity-70")
        lock_icon = "lock_open" if is_authenticated() else "lock"
        lock_color = "positive" if is_authenticated() else ""
        lock_btn = ui.button(icon=lock_icon, on_click=_toggle_auth).props(
            f"flat round color={lock_color}"
        )

    def _update_clocks() -> None:
        now = datetime.now(timezone.utc)
        ut_label.text  = f"UT  {now.strftime('%H:%M:%S')}"
        lst_label.text = f"LST {_fmt_hms(_calc_lst(now))}"

    ui.timer(1.0, _update_clocks)
    _update_clocks()

    with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
        for instr_id in mgr.instrument_ids:
            panels.append(AnalyzerPanel(instr_id))

        # Applica stato auth iniziale e avvia display timer se strumento già connesso
        for p in panels:
            p.apply_auth(is_authenticated())
            if mgr.is_connected(p.instr_id):
                p._set_connected(True)

        if len(panels) == 2:
            ComparisonPanel(panels)

        AntennaPanel() 
# ── Avvio ─────────────────────────────────────────────────

if __name__ in {"__main__", "__mp_main__"}:
    app.on_startup(lambda: asyncio.create_task(
        ServManager.antenna_monitor(SERV_HOST, SERVPORT)
    ))
    ui.run(
        title="Monitoring Parabola",
        host="0.0.0.0",
        port=8080,
        reload=False,       # True durante lo sviluppo
        storage_secret=STORAGE_SECRET,
    )
