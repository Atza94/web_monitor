"""
InstrumentManager – layer thread-safe tra PyVISA e il resto dell'app.

Ogni strumento ha il proprio lock: le chiamate SCPI vengono serializzate
per strumento, ma i due analizzatori possono essere interrogati in parallelo.
"""
from __future__ import annotations
import asyncio
import socket
import numpy as np
import pyvisa
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional
from config import INSTRUMENTS, VISA_TIMEOUT_MS

logger = logging.getLogger(__name__)
reader: asyncio.StreamReader
writer: asyncio.StreamWriter

# ── Data classes per stato antenna da serv.c ──────────────
@dataclass
class AntennaState:
    az_deg: float = 0.0
    el_deg: float = 0.0
    az_cmd: float = 0.0
    el_cmd: float = 0.0
    point_err: float = 0.0
    on_source: int = 0
    name_src: str = "?"
    rx_type: str = "?"
    lo_mhz: float = 0.0
    noise_cal: int = 0
    sub_act: list[float] = field(default_factory=lambda: [0.0] * 5)
    sub_cmd: list[float] = field(default_factory=lambda: [0.0] * 5)
    sub_mode: int = 0
    temp: float = 0.0
    pres: float = 0.0
    hum: float = 0.0
    wind: float = 0.0
# ── Data classes per i risultati ──────────────────────────

@dataclass
class TraceData:
    """Risultato di una lettura traccia."""
    frequencies: np.ndarray   # asse X in Hz
    amplitudes: np.ndarray    # asse Y in dBm
    start_freq: float
    stop_freq: float
    num_points: int


@dataclass
class InstrumentParams:
    """Stato corrente dei parametri dell'analizzatore."""
    center_freq_hz: float = 0.0
    span_hz: float = 0.0
    start_freq_hz: float = 0.0
    stop_freq_hz: float = 0.0
    rbw_hz: float = 0.0
    vbw_hz: float = 0.0
    sweep_time_s: float = 0.0

@dataclass
class InstrumentHandle:
    """Stato interno per un singolo strumento."""
    visa_address: str
    label: str
    resource: Optional[pyvisa.resources.MessageBasedResource] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    connected: bool = False


class InstrumentManager:
    """
    Gestisce le connessioni VISA e serializza gli accessi.

    Uso tipico:
        mgr = InstrumentManager()
        mgr.connect("DSA1030_1")
        params = mgr.get_params("DSA1030_1")
        trace  = mgr.read_trace("DSA1030_1")
    """

    def __init__(self) -> None:
        self._rm = pyvisa.ResourceManager("@py")  # usa pyvisa-py backend
        self._instruments: dict[str, InstrumentHandle] = {}

        for instr_id, cfg in INSTRUMENTS.items():
            self._instruments[instr_id] = InstrumentHandle(
                visa_address=cfg["visa_address"],
                label=cfg["label"],
            )

    @property
    def instrument_ids(self) -> list[str]:
        return list(self._instruments.keys())

    def get_label(self, instr_id: str) -> str:
        return self._instruments[instr_id].label

    def is_connected(self, instr_id: str) -> bool:
        return self._instruments[instr_id].connected

    # ── Connessione / Disconnessione ──────────────────────

    def connect(self, instr_id: str) -> None:
        handle = self._instruments[instr_id]
        with handle.lock:
            if handle.connected:
                return
            try:
                res = self._rm.open_resource(handle.visa_address)
                res.timeout = VISA_TIMEOUT_MS
                idn = res.query("*IDN?").strip()
                logger.info("Connesso a %s: %s", instr_id, idn)
                handle.resource = res
                handle.connected = True
            except pyvisa.Error as exc:
                logger.error("Connessione fallita per %s: %s", instr_id, exc)
                raise ConnectionError(
                    f"Impossibile connettersi a {handle.label} ({handle.visa_address})"
                ) from exc

    def disconnect(self, instr_id: str) -> None:
        handle = self._instruments[instr_id]
        with handle.lock:
            if handle.resource is not None:
                try:
                    handle.resource.close()
                except pyvisa.Error:
                    pass
            handle.resource = None
            handle.connected = False
            logger.info("Disconnesso da %s", instr_id)

    def disconnect_all(self) -> None:
        for instr_id in self._instruments:
            self.disconnect(instr_id)

    # ── Comandi SCPI interni ──────────────────────────────

    def _write(self, handle: InstrumentHandle, cmd: str) -> None:
        """Invia un comando SCPI (lock deve essere già acquisito)."""
        handle.resource.write(cmd)

    def _query(self, handle: InstrumentHandle, cmd: str) -> str:
        """Invia una query SCPI e ritorna la risposta (lock già acquisito)."""
        return handle.resource.query(cmd).strip()

    def _query_float(self, handle: InstrumentHandle, cmd: str) -> float:
        return float(self._query(handle, cmd))

    # ── Lettura parametri ─────────────────────────────────

    def get_params(self, instr_id: str) -> InstrumentParams:
        """Legge lo stato corrente dell'analizzatore."""
        handle = self._instruments[instr_id]
        with handle.lock:
            if not handle.connected:
                raise ConnectionError(f"{instr_id} non connesso")
            return InstrumentParams(
                center_freq_hz=self._query_float(handle, ":SENSe:FREQuency:CENTer?"),
                span_hz=self._query_float(handle, ":SENSe:FREQuency:SPAN?"),
                start_freq_hz=self._query_float(handle, ":SENSe:FREQuency:STARt?"),
                stop_freq_hz=self._query_float(handle, ":SENSe:FREQuency:STOP?"),
                rbw_hz=self._query_float(handle, ":SENSe:BANDwidth:RESolution?"),
                vbw_hz=self._query_float(handle, ":SENSe:BANDwidth:VIDeo?"),
                sweep_time_s=self._query_float(handle, ":SENSe:SWEep:TIME?"),
            )

    # ── Impostazione parametri ────────────────────────────

    def set_center_freq(self, instr_id: str, freq_hz: float) -> None:
        handle = self._instruments[instr_id]
        with handle.lock:
            self._write(handle, f":SENSe:FREQuency:CENTer {freq_hz:.0f}")

    def set_span(self, instr_id: str, span_hz: float) -> None:
        handle = self._instruments[instr_id]
        with handle.lock:
            self._write(handle, f":SENSe:FREQuency:SPAN {span_hz:.0f}")

    def set_rbw(self, instr_id: str, rbw_hz: float) -> None:
        handle = self._instruments[instr_id]
        with handle.lock:
            self._write(handle, f":SENSe:BANDwidth:RESolution {rbw_hz:.0f}")

    def set_vbw(self, instr_id: str, vbw_hz: float) -> None:
        handle = self._instruments[instr_id]
        with handle.lock:
            self._write(handle, f":SENSe:BANDwidth:VIDeo {vbw_hz:.0f}")

    def set_start_stop(self, instr_id: str, start_hz: float, stop_hz: float) -> None:
        handle = self._instruments[instr_id]
        with handle.lock:
            self._write(handle, f":SENSe:FREQuency:STARt {start_hz:.0f}")
            self._write(handle, f":SENSe:FREQuency:STOP {stop_hz:.0f}")

    def free_field(self, instr_id: str, command: str) -> str | None:
        command = command.strip()
        if not command:
            return None
        handle = self._instruments[instr_id]
        with handle.lock:
            if "?" in command:  # le query SCPI possono avere parametri dopo il '?' (es. :TRACe:MODE? TRACE1)
                return self._query(handle, command)
            else:
                self._write(handle, command)
                return None


    # ── Lettura traccia ───────────────────────────────────

    @staticmethod
    def _strip_block_header(raw: str) -> str:
        """
        Rimuove l'header IEEE 488.2 definite length block (#NXXX...),
        se presente.  Es: '#9000045014 -1.49e+01,...' → '-1.49e+01,...'
        """
        if not raw.startswith("#"):
            return raw
        # Il carattere dopo '#' indica quante cifre compongono il campo lunghezza
        n_digits = int(raw[1])
        # L'header è: '#' + 1 cifra + n_digits cifre = 2 + n_digits caratteri
        header_len = 2 + n_digits
        return raw[header_len:].lstrip()

    def read_trace(self, instr_id: str, trace_n: int = 1) -> TraceData:
        """
        Legge la traccia corrente dal DSA1030.
        Il DSA1030 ritorna i dati con header IEEE 488.2 (#NXXX...)
        seguito da valori ASCII separati da virgola.
        """
        handle = self._instruments[instr_id]
        with handle.lock:
            if not handle.connected:
                raise ConnectionError(f"{instr_id} non connesso")

            # Leggi estremi frequenza per costruire l'asse X
            start = self._query_float(handle, ":SENSe:FREQuency:STARt?")
            stop = self._query_float(handle, ":SENSe:FREQuency:STOP?")

            # Leggi dati traccia – strip header IEEE 488.2
            raw = self._query(handle, f":TRACe:DATA? TRACE{trace_n}")
            raw = self._strip_block_header(raw)
            amplitudes = np.array([float(v) for v in raw.split(",")])

            num_points = len(amplitudes)
            frequencies = np.linspace(start, stop, num_points)

            return TraceData(
                frequencies=frequencies,
                amplitudes=amplitudes,
                start_freq=start,
                stop_freq=stop,
                num_points=num_points,
            )

class ServManager:
    """
    Gestisce la connessione al servizio hostato su Workstation Field System.

    Il servizio risponde a richieste TCP con Array che descrivono
    lo stato corrente dell'antenna (posizione, frequenza, ecc).
    """
    state = AntennaState()
    connected: bool = False
    @staticmethod
    async def _query_async(reader: asyncio.StreamReader,
                           writer: asyncio.StreamWriter,
                           keyword: str) -> list[str]:
        writer.write((keyword + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        return line.decode().split()[1:]  # scarta prefisso lunghezza

    @staticmethod
    async def antenna_monitor(host: str, port: int,
                              fast_interval: float = 2.0, slow_every: int = 5):
        tick = 0
        while True:
            try:
                reader, writer = await asyncio.open_connection(host, port)
                ServManager.connected = True
                logger.info("Connesso a serv su %s:%d", host, port)
                try:
                    while True:
                        await ServManager._update_fast(reader, writer)
                        if tick % slow_every == 0:
                            await ServManager._update_slow(reader, writer)
                        tick += 1
                        await asyncio.sleep(fast_interval)
                finally:
                    ServManager.connected = False
                    writer.close()
                    await writer.wait_closed()
            except (OSError, asyncio.IncompleteReadError, asyncio.TimeoutError) as exc:
                ServManager.connected = False
                logger.warning("serv disconnesso: %s — retry in 10 s", exc)
                tick = 0
                await asyncio.sleep(10)

    @staticmethod
    async def _update_fast(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        t = await ServManager._query_async(reader, writer, "fupdate")
        # azc elc azp elp daz del pnterr ionsor incyc  (dopo strip prefisso)
        # [0]  [1] [2] [3] [4] [5] [6]    [7]    [8]
        ServManager.state.az_cmd    = float(t[0])
        ServManager.state.el_cmd    = float(t[1])
        ServManager.state.az_deg    = float(t[2])
        ServManager.state.el_deg    = float(t[3])
        ServManager.state.point_err = float(t[6])
        ServManager.state.on_source = int(t[7])

        w = await ServManager._query_async(reader, writer, "ska")
        ServManager.state.wind = float(w[-3])



    @staticmethod
    async def _update_slow(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # ncal rtype lofreq loampl lostat rtemp rvac T P H
        # [0]  [1]   [2]    [3]    [4]    [5]   [6] [7][8][9]
        r = await ServManager._query_async(reader, writer, "updtrec")
        ServManager.state.noise_cal = int(r[0])
        ServManager.state.rx_type   = r[1]
        ServManager.state.lo_mhz    = float(r[2])
        ServManager.state.temp      = float(r[7])
        ServManager.state.pres      = float(r[8])
        ServManager.state.hum       = float(r[9])

        # cmd1-5 act1-5 mode rtype scust
        # [0..4] [5..9] [10] [11]  [12]
        s = await ServManager._query_async(reader, writer, "updtsub")
        ServManager.state.sub_cmd  = [float(s[i]) for i in range(0, 5)]
        ServManager.state.sub_act  = [float(s[i]) for i in range(5, 10)]
        ServManager.state.sub_mode = int(s[10])

        v = await ServManager._query_async(reader, writer, "updsrce")
        ServManager.state.name_src = v[0]


