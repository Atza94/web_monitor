# Spectrum Web

Interfaccia web per il controllo e la visualizzazione live di due Rigol DSA1030
presso la Stazione Radioastronomica di Medicina.

## Struttura

```
spectrum_web/
├── main.py           # App NiceGUI (UI + pagina web)
├── instruments.py    # InstrumentManager thread-safe (PyVISA)
├── config.py         # Indirizzi VISA, parametri di default
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Modifica `config.py` con gli indirizzi IP corretti dei tuoi DSA1030:
```python
INSTRUMENTS = {
    "DSA1030_1": {
        "visa_address": "TCPIP0::192.168.1.100::INSTR",
        "label": "DSA1030 #1",
    },
    ...
}
```

## Avvio

```bash
python main.py
```

Apri il browser su `http://localhost:8080` (o l'IP della macchina sulla LAN).

## Architettura

- **`InstrumentManager`** – Singleton che gestisce le connessioni VISA. Ogni
  strumento ha il proprio `threading.Lock`, quindi le letture dei due
  analizzatori possono avvenire in parallelo, ma gli accessi allo stesso
  strumento sono serializzati.

- **`AnalyzerPanel`** – Widget NiceGUI composito che crea la UI per un singolo
  strumento: grafico Plotly con aggiornamento periodico via `ui.timer`, e
  controlli per center freq / span / RBW / VBW.

- Il layer web (NiceGUI) non tocca mai PyVISA direttamente: passa sempre
  dall'`InstrumentManager`. Questo rende possibile in futuro esporre lo
  stesso manager via FastAPI REST, test con mock, ecc.

## Note

- Il DSA1030 ha 601 punti per traccia di default.
- Il polling è configurabile in `config.py` (`TRACE_UPDATE_INTERVAL_S`).
  Con 1s di intervallo il carico sulla LAN è trascurabile.
- Per sviluppo senza strumenti fisici, puoi creare un mock di
  `InstrumentManager` che ritorna dati casuali.
