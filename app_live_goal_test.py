import os
import math
import sqlite3
import zipfile
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Candidate Gol 2T", page_icon="⚽", layout="wide")

ITALY_TZ = ZoneInfo("Europe/Rome")
API_BASE = "https://api.football-data-api.com"

O15_MIN = 0.84
O25_MIN = 0.60
LIVE_MIN = 0.80

# Paniere concordato: nessun selettore, queste competizioni sono sempre attive.
SELECTED_LEAGUES = {
    'England Premier League',
    'Spain La Liga',
    'Italy Serie A',
    'Germany Bundesliga',
    'Germany 2. Bundesliga',
    'France Ligue 1',
    'Portugal Liga NOS',
    'Netherlands Eredivisie',
    'Netherlands Eerste Divisie',
    'Belgium Pro League',
}

CANDIDATE_K = [300, 500, 750, 1000, 1500, 2000, 3000, 5000]
MIN_K = 300
MAX_STABILITY_DELTA = 0.025
MAX_DISTANCE_GROWTH = 1.30


st.markdown(
    """
    <style>
    .block-container {max-width: 1150px; padding-top: 1.4rem;}
    .goal2t-title {font-size: 1.8rem; font-weight: 800; margin-bottom: .1rem;}
    .goal2t-sub {color: #6b7280; margin-bottom: 1rem;}
    [data-testid="stDataFrame"] {border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key():
    try:
        value = st.secrets.get("FOOTYSTATS_API_KEY") or st.secrets.get("API_KEY")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return (os.getenv("FOOTYSTATS_API_KEY") or os.getenv("API_KEY") or "").strip()



# CONFIGURAZIONE DB PER STREAMLIT CLOUD
# Il database ottimizzato (352.557 gare) e' incluso in v13_historical.zip.
# Non sovrascrivere il vecchio .db del repository: lo ZIP aggiornato ha priorita'.
V13_DATABASE_MIN_MATCHES = 352557
APP_DIR = Path(__file__).resolve().parent


def _configured_v13_db():
    try:
        value = st.secrets.get("V13_DB_PATH")
    except Exception:
        value = None
    value = value or os.environ.get("V13_DB_PATH")
    if value:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else APP_DIR / path
    return APP_DIR / "v13_historical.db"


def _v13_db_count(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Database mancante: {path}")
    with sqlite3.connect(str(path)) as connection:
        fields = {r[1] for r in connection.execute("PRAGMA table_info(matches_v13)")}
        needed = {"match_id", "competition_name", "season_id", "date_unix",
                  "status", "result_1x2", "home_goals", "away_goals",
                  "ht_goals_team_a", "ht_goals_team_b", "odds_ft_over15",
                  "odds_ft_under15", "odds_ft_over25", "odds_ft_under25"}
        missing = sorted(needed - fields)
        if missing:
            raise RuntimeError("Colonne mancanti in matches_v13: " + ", ".join(missing))
        return int(connection.execute("SELECT COUNT(*) FROM matches_v13").fetchone()[0])


def _v13_archive_options():
    found = []
    for archive in sorted(APP_DIR.glob("v13_historical*.zip")):
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                info = next((m for m in zf.infolist()
                             if Path(m.filename).name == "v13_historical.db"), None)
                if info:
                    found.append((archive, info.filename, info.file_size, info.CRC))
        except (zipfile.BadZipFile, OSError):
            continue
    return sorted(found, key=lambda item: item[2], reverse=True)


def _v13_extract_archive(archive, member, expected_size, crc):
    # Streamlit Cloud ha disco scrivibile temporaneo; resta valido tra i rerun.
    cache = Path(tempfile.gettempdir()) / "v13_streamlit_history"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"v13_{expected_size}_{crc:08x}.db"
    if target.is_file():
        try:
            return target, _v13_db_count(target)
        except Exception:
            target.unlink(missing_ok=True)
    partial = cache / (target.name + ".partial")
    try:
        with st.spinner("Prima apertura: preparo il database V13 aggiornato..."):
            with zipfile.ZipFile(archive, "r") as zf:
                with zf.open(member) as source, partial.open("wb") as dest:
                    shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
        if partial.stat().st_size != expected_size:
            raise RuntimeError("Estrazione incompleta del database V13")
        count = _v13_db_count(partial)
        os.replace(partial, target)
        return target, count
    finally:
        partial.unlink(missing_ok=True)


def _initialize_v13_db():
    db_path = _configured_v13_db()
    explicit = bool(os.environ.get("V13_DB_PATH"))
    try:
        explicit = explicit or bool(st.secrets.get("V13_DB_PATH"))
    except Exception:
        pass
    # Se l'utente ha configurato un percorso preciso, rispettalo.
    if explicit and db_path.is_file():
        count = _v13_db_count(db_path)
        if count < V13_DATABASE_MIN_MATCHES:
            st.warning(f"Database esterno ancora vecchio: {count:,} gare")
        return db_path, count

    archives = _v13_archive_options()
    # Priorita' al nuovo ZIP: sostituisce logicamente l'archivio vecchio,
    # senza cancellare o modificare il file gia' presente su GitHub.
    for archive, member, unpacked_bytes, crc in archives:
        if unpacked_bytes < 60_000_000:
            continue
        db, count = _v13_extract_archive(archive, member, unpacked_bytes, crc)
        if count >= V13_DATABASE_MIN_MATCHES:
            return db, count

    # Compatibilita' durante la migrazione: il vecchio software resta attivo
    # fino a quando non viene caricato il nuovo v13_historical.zip.
    if db_path.is_file():
        count = _v13_db_count(db_path)
        if count < V13_DATABASE_MIN_MATCHES:
            st.warning(f"Archivio precedente ({count:,} gare). Carica v13_historical.zip aggiornato sul repository per passare automaticamente a 352.557 gare.")
        return db_path, count
    if archives:
        archive, member, size, crc = archives[0]
        db, count = _v13_extract_archive(archive, member, size, crc)
        if count < V13_DATABASE_MIN_MATCHES:
            st.warning(f"Archivio precedente ({count:,} gare). Carica il nuovo v13_historical.zip per aggiornare automaticamente i dati.")
        return db, count
    raise FileNotFoundError("Nessun v13_historical.db o v13_historical.zip disponibile")


def _v13_db_fingerprint(path):
    info = Path(path).stat()
    return info.st_size, info.st_mtime_ns


API_KEY = get_api_key()
try:
    DB_FILE, V13_DB_ROWS = _initialize_v13_db()
except Exception as _db_exc:
    st.error(f"Database V13: {_db_exc}")
    st.stop()


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def pair_prob(over_odd, under_odd):
    a, b = num(over_odd), num(under_odd)
    if not np.isfinite(a) or not np.isfinite(b) or a <= 1 or b <= 1:
        return np.nan
    ia, ib = 1.0 / a, 1.0 / b
    return ia / (ia + ib)


# Mediana dell'overround O/U osservata nel DB storico (~6,67%).
# Viene usata SOLO se FootyStats fornisce un solo lato della quota O/U.
OU_FALLBACK_OVERROUND = 1.0666666666666667


def ou_prob(over_odd, under_odd):
    """Probabilita' Over de-marginalizzata; tollera un solo lato quota mancante."""
    a, b = num(over_odd), num(under_odd)

    # Caso normale: entrambe le quote presenti -> logica originale IDENTICA.
    if np.isfinite(a) and np.isfinite(b) and a > 1 and b > 1:
        ia, ib = 1.0 / a, 1.0 / b
        return ia / (ia + ib)

    # Fallback prudente: una sola quota disponibile.
    if np.isfinite(a) and a > 1:
        return float(np.clip((1.0 / a) / OU_FALLBACK_OVERROUND, 0.0, 1.0))
    if np.isfinite(b) and b > 1:
        p_under = (1.0 / b) / OU_FALLBACK_OVERROUND
        return float(np.clip(1.0 - p_under, 0.0, 1.0))

    return np.nan


def probs_1x2(a, b, c):
    vals = np.array([num(a), num(b), num(c)], dtype=float)
    if np.any(~np.isfinite(vals)) or np.any(vals <= 1):
        return [np.nan, np.nan, np.nan]
    inv = 1.0 / vals
    return (inv / inv.sum()).tolist()


def feature(d):
    p1, px, p2 = probs_1x2(d.get("odds_ft_1"), d.get("odds_ft_x"), d.get("odds_ft_2"))
    return np.array(
        [
            p1,
            px,
            p2,
            ou_prob(d.get("odds_ft_over15"), d.get("odds_ft_under15")),
            ou_prob(d.get("odds_ft_over25"), d.get("odds_ft_under25")),
            ou_prob(d.get("odds_ft_over35"), d.get("odds_ft_under35")),
            pair_prob(d.get("odds_btts_yes"), d.get("odds_btts_no")),
        ],
        dtype=float,
    )


def robust_scale(X):
    med = np.nanmedian(X, axis=0)
    iqr = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
    iqr[(~np.isfinite(iqr)) | (iqr < 1e-9)] = 1.0
    med[~np.isfinite(med)] = 0.0
    return med, iqr, (X - med) / iqr


def probabilities_for_positions(pos, dist, acts, markets):
    ds = dist[pos]
    w = 1.0 / (ds + 0.15)
    den = w.sum()
    p = {}
    for market in markets:
        vals = np.array([acts[i][market] for i in pos], dtype=float)
        p[market] = float(np.sum(w * vals) / den)
    return p, ds


def adaptive_market_sample(dist, acts, markets):
    order = np.argsort(dist)
    available = len(order)
    if available == 0:
        raise RuntimeError("Nessun affine storico disponibile.")

    initial_k = min(MIN_K, available)
    chosen_k = initial_k
    chosen_p, chosen_ds = probabilities_for_positions(order[:chosen_k], dist, acts, markets)
    prev_p = chosen_p
    prev_max_dist = float(chosen_ds[-1])

    for k in [x for x in CANDIDATE_K if initial_k < x <= available]:
        p, ds = probabilities_for_positions(order[:k], dist, acts, markets)
        max_dist = float(ds[-1])
        delta = max(abs(p[m] - prev_p[m]) for m in markets)
        growth = max_dist / prev_max_dist if prev_max_dist > 0 else 1.0
        if delta > MAX_STABILITY_DELTA or growth > MAX_DISTANCE_GROWTH:
            break
        chosen_k = k
        chosen_p = p
        chosen_ds = ds
        prev_p = p
        prev_max_dist = max_dist

    return chosen_p, chosen_ds, order[:chosen_k]


@st.cache_resource(show_spinner=False)
def load_live45_history(db_path_str, db_fingerprint):
    """Precalcola una sola volta le caratteristiche e gli esiti per i 3 HT.

    L'algoritmo Affini (normalizzazione, distanze, K adattivo, pesi) resta
    quello precedente; lo rendiamo vettoriale per evitare di scandire lo
    stesso archivio 3 volte per ogni partita della giornata.
    """
    db_path = Path(db_path_str)
    if not db_path.is_file():
        return None

    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    cols = [
        "match_id", "date_unix", "home_goals", "away_goals",
        "ht_goals_team_a", "ht_goals_team_b",
        "odds_ft_1", "odds_ft_x", "odds_ft_2",
        "odds_ft_over15", "odds_ft_under15",
        "odds_ft_over25", "odds_ft_under25",
        "odds_ft_over35", "odds_ft_under35",
        "odds_btts_yes", "odds_btts_no",
    ]
    try:
        rows = [dict(r) for r in con.execute(
            f"SELECT {','.join(cols)} FROM matches_v13 "
            "WHERE status='complete' AND result_1x2 IN ('1','X','2') "
            "AND ht_goals_team_a IS NOT NULL AND ht_goals_team_b IS NOT NULL "
            "AND date_unix < CAST(strftime('%s','now') AS INTEGER) "
            "ORDER BY date_unix, match_id"
        )]
    finally:
        con.close()
    if not rows:
        return None

    raw = np.vstack([feature(r) for r in rows])
    valid = np.all(np.isfinite(raw), axis=1)
    rows = [r for r, ok in zip(rows, valid) if ok]
    raw = raw[valid]
    if len(rows) == 0:
        return None
    med, iqr, Z = robust_scale(raw)
    n = len(rows)
    ht_h = np.fromiter((int(r["ht_goals_team_a"]) for r in rows), dtype=np.int16, count=n)
    ht_a = np.fromiter((int(r["ht_goals_team_b"]) for r in rows), dtype=np.int16, count=n)
    hg = np.fromiter((int(r["home_goals"]) for r in rows), dtype=np.int16, count=n)
    ag = np.fromiter((int(r["away_goals"]) for r in rows), dtype=np.int16, count=n)
    dates = np.fromiter((int(r["date_unix"]) for r in rows), dtype=np.int64, count=n)
    # Sette mercati usati per decidere l'ampiezza del campione adattivo V13.
    h2, a2 = np.maximum(hg - ht_h, 0), np.maximum(ag - ht_a, 0)
    outcomes = np.column_stack((
        hg > ag, hg == ag, hg < ag, (hg + ag) >= 3,
        (hg >= 1) & (ag >= 1), h2 >= 1, a2 >= 1,
        (h2 + a2) >= 1,  # ottava colonna: almeno un gol nel secondo tempo
    )).astype(np.float64)
    return Z, med, iqr, ht_h, ht_a, dates, outcomes


def live45_actuals(r):
    hg = int(r["home_goals"])
    ag = int(r["away_goals"])
    hh = int(r["ht_goals_team_a"])
    ah = int(r["ht_goals_team_b"])
    h2 = max(0, hg - hh)
    a2 = max(0, ag - ah)
    total = hg + ag
    return {
        "1": float(hg > ag),
        "X": float(hg == ag),
        "2": float(hg < ag),
        "Gol nel 2T": float(h2 + a2 >= 1),
        "Over 0.5 FT": float(total >= 1),
        "Over 2.5 FT": float(total >= 3),
        "Gol FT": float(hg >= 1 and ag >= 1),
        "Casa segna ancora": float(h2 >= 1),
        "Trasferta segna ancora": float(a2 >= 1),
    }


def second_half_goal_probability(match, ht_home, ht_away):
    """Probabilita' di almeno 1 gol nel 2T, condizionata al risultato HT.

    Algoritmo V13 in forma vettoriale: nessun dato futuro rispetto al target,
    scala robusta dell'archivio completo, distanze RMS fra quote,
    campione 300/500/... adattivo con delta <=2,5 pp e crescita <=1,30,
    e media pesata 1/(distanza+0,15).
    """
    loaded = load_live45_history(str(DB_FILE), _v13_db_fingerprint(DB_FILE))
    if loaded is None:
        return np.nan
    Z, med, iqr, ht_h, ht_a, dates, outcomes = loaded
    raw_date = num(match.get("date_unix"))
    cutoff = int(raw_date) if np.isfinite(raw_date) else int(datetime.now().timestamp())
    idx = np.flatnonzero((ht_h == int(ht_home)) & (ht_a == int(ht_away)) & (dates < cutoff))
    if len(idx) < 30:
        return np.nan

    target = feature(match)
    if not np.all(np.isfinite(target)):
        return np.nan
    zt = (target - med) / iqr
    ds = np.sqrt(np.mean(np.square(Z[idx] - zt), axis=1))
    order = np.argsort(ds)
    available = len(order)
    first_k = min(MIN_K, available)

    def summary(k):
        chosen = order[:k]
        w = 1.0 / (ds[chosen] + 0.15)
        prob = np.average(outcomes[idx[chosen]], axis=0, weights=w)
        return prob, float(ds[chosen][-1])

    chosen_k = first_k
    previous, previous_max_dist = summary(first_k)
    for k in [x for x in CANDIDATE_K if first_k < x <= available]:
        current, max_dist = summary(k)
        delta = float(np.max(np.abs(current[:7] - previous[:7])))
        growth = max_dist / previous_max_dist if previous_max_dist > 0 else 1.0
        if delta > MAX_STABILITY_DELTA or growth > MAX_DISTANCE_GROWTH:
            break
        chosen_k = k
        previous, previous_max_dist = current, max_dist
    return float(previous[7])


@st.cache_data(show_spinner=False)
def load_league_map(db_path_str, db_fingerprint):
    con = sqlite3.connect(db_path_str)
    con.row_factory = sqlite3.Row
    league_map = {}
    try:
        for r in con.execute(
            """
            SELECT season_id, MAX(competition_name) AS competition_name
            FROM matches_v13
            WHERE competition_name IS NOT NULL
            GROUP BY season_id
            """
        ):
            league_map[int(r["season_id"])] = str(r["competition_name"])
    finally:
        con.close()
    return league_map


@st.cache_data(ttl=600, show_spinner=False)
def get_day_matches(day_iso):
    params = {"key": API_KEY, "date": day_iso, "timezone": "Europe/Rome", "page": 1}
    out = []
    while True:
        response = requests.get(API_BASE + "/todays-matches", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        if not isinstance(data, list):
            data = []
        out.extend(data)

        pager = payload.get("pager") or {}
        current = int(pager.get("current_page") or pager.get("page") or params["page"])
        maximum = int(pager.get("max_page") or pager.get("total_pages") or current)
        if current >= maximum or not data:
            break
        params["page"] = current + 1

    deduped = {}
    for m in out:
        key = m.get("id") or m.get("match_id") or (m.get("home_name"), m.get("away_name"), m.get("date_unix"))
        deduped[str(key)] = m
    return list(deduped.values())


@st.cache_data(ttl=600, show_spinner=False)
def get_match_detail(match_id):
    response = requests.get(
        API_BASE + "/match",
        params={"key": API_KEY, "match_id": int(match_id)},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", payload)


def fixture_league_name_fast(m, league_map):
    sid = m.get("competition_id") or m.get("season_id")
    try:
        sid = int(sid)
    except Exception:
        sid = None
    if sid is not None and sid in league_map:
        return str(league_map[sid])
    for key in ("_league_name", "competition_name", "league_name", "league"):
        value = m.get(key)
        if value not in (None, "", "None"):
            return str(value)
    return None


def enrich_match(m, league_map):
    mid = m.get("id") or m.get("match_id")
    merged = dict(m)
    if mid:
        try:
            detail = get_match_detail(int(mid))
            if isinstance(detail, dict):
                merged.update(detail)
        except Exception:
            pass

    sid = merged.get("competition_id") or merged.get("season_id")
    try:
        sid = int(sid)
    except Exception:
        sid = None

    if sid is not None and sid in league_map:
        merged["_league_name"] = league_map[sid]
    else:
        merged["_league_name"] = fixture_league_name_fast(merged, league_map) or "Campionato non identificato"
    return merged


def team_name(m, side):
    if side == "home":
        return str(m.get("home_name") or m.get("homeName") or m.get("team_a_name") or "Casa")
    return str(m.get("away_name") or m.get("awayName") or m.get("team_b_name") or "Trasferta")


def league_name(m):
    return str(m.get("_league_name") or m.get("competition_name") or "Campionato non identificato")


def pct_text(value):
    if not np.isfinite(num(value)):
        return "n/d"
    return f"{float(value) * 100:.1f}%"


def build_candidates(chosen_day):
    league_map = load_league_map(str(DB_FILE), _v13_db_fingerprint(DB_FILE))
    fixtures = get_day_matches(chosen_day.isoformat())

    # Prima identifichiamo le sole competizioni ammesse, cosi evitiamo chiamate /match inutili.
    selected = []
    unresolved = []
    for fixture in fixtures:
        lg = fixture_league_name_fast(fixture, league_map)
        if lg is None:
            unresolved.append(fixture)
        elif lg in SELECTED_LEAGUES:
            selected.append(fixture)

    rows = []
    for fixture in selected + unresolved:
        fast_lg = fixture_league_name_fast(fixture, league_map)
        if fast_lg is not None and fast_lg not in SELECTED_LEAGUES:
            continue

        match = enrich_match(fixture, league_map)
        lg = league_name(match)
        if lg not in SELECTED_LEAGUES:
            continue

        p_o15 = ou_prob(match.get("odds_ft_over15"), match.get("odds_ft_under15"))
        p_o25 = ou_prob(match.get("odds_ft_over25"), match.get("odds_ft_under25"))
        if not (np.isfinite(p_o15) and np.isfinite(p_o25)):
            continue
        if p_o15 < O15_MIN or p_o25 < O25_MIN:
            continue

        scores = [(0, 0), (1, 0), (0, 1)]
        live_probs = {
            ht: second_half_goal_probability(match, ht[0], ht[1])
            for ht in scores
        }
        qualifying = [
            f"{ht[0]}-{ht[1]}" for ht, p in live_probs.items()
            if np.isfinite(p) and p >= LIVE_MIN
        ]
        rows.append(
            {
                "Campionato": lg,
                "Partita": f"{team_name(match, 'home')} - {team_name(match, 'away')}",
                "O1.5 pre": pct_text(p_o15),
                "O2.5 pre": pct_text(p_o25),
                "Gol 2T se HT 0-0": pct_text(live_probs[(0, 0)]),
                "Gol 2T se HT 1-0": pct_text(live_probs[(1, 0)]),
                "Gol 2T se HT 0-1": pct_text(live_probs[(0, 1)]),
                "HT con LIVE >=80%": ", ".join(qualifying) if qualifying else "Nessuno",
                "_date_unix": int(num(match.get("date_unix"))) if np.isfinite(num(match.get("date_unix"))) else 0,
            }
        )

    rows.sort(key=lambda x: (x["_date_unix"], x["Campionato"], x["Partita"]))
    for row in rows:
        row.pop("_date_unix", None)
    return rows, len(fixtures)


def main():
    st.markdown('<div class="goal2t-title">Candidate Gol 2° Tempo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="goal2t-sub">10 leghe · O1.5 pre ≥ 84% · O2.5 pre ≥ 60% · Affini LIVE ≥ 80% · HT 0-0 / 1-0 / 0-1</div>',
        unsafe_allow_html=True,
    )

    if not API_KEY:
        st.error("API key FootyStats non trovata. Inserisci FOOTYSTATS_API_KEY nei Secrets di Streamlit o nelle variabili ambiente.")
        st.stop()

    if not DB_FILE.exists():
        st.error(
            f"Non trovo il database storico: {DB_FILE}. "
            "Carica v13_historical.db oppure v13_historical.zip nella stessa cartella dell'app, "
            "oppure imposta V13_DB_PATH."
        )
        st.stop()

    left, right = st.columns([1.5, 0.8])
    with left:
        chosen_day = st.date_input("Data", value=date.today())
    with right:
        st.write("")
        st.write("")
        if st.button("Aggiorna dati", use_container_width=True):
            get_day_matches.clear()
            get_match_detail.clear()
            load_live45_history.clear()
            load_league_map.clear()
            st.rerun()

    try:
        with st.spinner("Carico e filtro le partite della giornata..."):
            candidates, total_fixtures = build_candidates(chosen_day)
    except requests.HTTPError as exc:
        st.error(f"Errore FootyStats: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Errore durante il caricamento: {exc}")
        st.stop()

    st.markdown(f"### {len(candidates)} partite candidate")
    st.caption(
        f"Archivio V13: {V13_DB_ROWS:,} partite · FootyStats oggi: {total_fixtures}. "
        "Le percentuali LIVE sono ipotesi calcolate per ciascun risultato HT; "
        "considera valida soltanto la colonna corrispondente al risultato reale all'intervallo "
        "e solo se almeno 80%."
    )

    if not candidates:
        st.info("Nessuna partita della giornata rispetta contemporaneamente O1.5 ≥84% e O2.5 ≥60%.")
    else:
        df = pd.DataFrame(candidates)
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Campionato": st.column_config.TextColumn(width="medium"),
                "Partita": st.column_config.TextColumn(width="large"),
                "O1.5 pre": st.column_config.TextColumn(width="small"),
                "O2.5 pre": st.column_config.TextColumn(width="small"),
                "Gol 2T se HT 0-0": st.column_config.TextColumn(width="medium"),
                "Gol 2T se HT 1-0": st.column_config.TextColumn(width="medium"),
                "Gol 2T se HT 0-1": st.column_config.TextColumn(width="medium"),
                "HT con LIVE >=80%": st.column_config.TextColumn(width="medium"),
            },
        )

    with st.expander("Competizioni incluse"):
        st.write(" · ".join(sorted(SELECTED_LEAGUES)))


if __name__ == "__main__":
    main()