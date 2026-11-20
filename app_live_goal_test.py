import os
import math
import sqlite3
import zipfile
import shutil
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

O15_MIN = 0.80
O25_MIN = 0.60

# Paniere concordato: nessun selettore, queste competizioni sono sempre attive.
SELECTED_LEAGUES = {
    "England Premier League",
    "Spain La Liga",
    "Italy Serie A",
    "France Ligue 1",
    "Portugal Liga NOS",
    "Turkey Süper Lig",
    "Norway Eliteserien",
    "Norway First Division",
    "Sweden Allsvenskan",
    "Czech Republic First League",
    "Europe UEFA Champions League",
    "Europe UEFA Europa League",
    "Europe UEFA Europa Conference League",
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


def resolve_db_path():
    try:
        configured = st.secrets.get("V13_DB_PATH")
        if configured:
            return Path(str(configured)).expanduser()
    except Exception:
        pass
    env_path = os.getenv("V13_DB_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return Path(__file__).with_name("v13_historical.db")


def ensure_db_available(db_path):
    """Estrae automaticamente v13_historical.db da v13_historical.zip se necessario."""
    db_path = Path(db_path)
    if db_path.exists():
        return db_path

    # Se V13_DB_PATH punta a un percorso personalizzato, cerca prima lo ZIP accanto al DB.
    zip_path = db_path.with_suffix(".zip")

    # Fallback: ZIP nella stessa cartella dell'app (caso tipico Streamlit Cloud).
    if not zip_path.exists():
        zip_path = Path(__file__).with_name("v13_historical.zip")

    if not zip_path.exists():
        return db_path

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            candidates = [
                name for name in zf.namelist()
                if Path(name).name == "v13_historical.db"
            ]
            if not candidates:
                raise RuntimeError(
                    f"Nel file {zip_path.name} non trovo v13_historical.db."
                )

            # Copia il DB nel percorso atteso dall'app, anche se nello ZIP e' dentro una sottocartella.
            with zf.open(candidates[0], "r") as source, db_path.open("wb") as target:
                shutil.copyfileobj(source, target)
    except Exception as exc:
        st.error(f"Errore nell'estrazione del database storico da {zip_path.name}: {exc}")
        return db_path

    return db_path


API_KEY = get_api_key()
DB_FILE = ensure_db_available(resolve_db_path())


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
            pair_prob(d.get("odds_ft_over15"), d.get("odds_ft_under15")),
            pair_prob(d.get("odds_ft_over25"), d.get("odds_ft_under25")),
            pair_prob(d.get("odds_ft_over35"), d.get("odds_ft_under35")),
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
def load_live45_history(db_path_str):
    db_path = Path(db_path_str)
    if not db_path.exists():
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
    rows = [dict(r) for r in con.execute(
        f"""
        SELECT {','.join(cols)}
        FROM matches_v13
        WHERE status='complete'
          AND result_1x2 IN ('1','X','2')
          AND ht_goals_team_a IS NOT NULL
          AND ht_goals_team_b IS NOT NULL
          AND date_unix < CAST(strftime('%s','now') AS INTEGER)
        ORDER BY date_unix, match_id
        """
    )]
    con.close()

    if not rows:
        return None

    X = np.vstack([feature(r) for r in rows])
    valid = np.all(np.isfinite(X), axis=1)
    rows = [r for r, ok in zip(rows, valid) if ok]
    X = X[valid]
    if not rows:
        return None

    med, iqr, Z = robust_scale(X)
    return rows, Z, med, iqr


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
        "Over 0.5 FT": float(total >= 1),
        "Over 2.5 FT": float(total >= 3),
        "Gol FT": float(hg >= 1 and ag >= 1),
        "Casa segna ancora": float(h2 >= 1),
        "Trasferta segna ancora": float(a2 >= 1),
    }


def over05_second_half_if_ht00(match):
    loaded = load_live45_history(str(DB_FILE))
    if not loaded:
        return np.nan

    rows, Z, med, iqr = loaded
    cutoff = int(num(match.get("date_unix"))) if np.isfinite(num(match.get("date_unix"))) else int(datetime.now().timestamp())
    idx = [
        i for i, r in enumerate(rows)
        if int(r["ht_goals_team_a"]) == 0
        and int(r["ht_goals_team_b"]) == 0
        and int(r["date_unix"]) < cutoff
    ]
    if len(idx) < 30:
        return np.nan

    target = feature(match)
    if not np.all(np.isfinite(target)):
        return np.nan

    zt = (target - med) / iqr
    Zi = Z[idx]
    dist = np.sqrt(np.mean(np.square(Zi - zt), axis=1))
    subset = [rows[i] for i in idx]
    acts = [live45_actuals(r) for r in subset]

    stability_markets = [
        "1", "X", "2", "Over 2.5 FT", "Gol FT",
        "Casa segna ancora", "Trasferta segna ancora",
    ]
    _, _, chosen = adaptive_market_sample(dist, acts, stability_markets)
    all_markets = list(acts[0].keys())
    probabilities, _ = probabilities_for_positions(chosen, dist, acts, all_markets)
    # Con HT 0-0, Over 0.5 FT equivale ad almeno un gol nel secondo tempo.
    return probabilities.get("Over 0.5 FT", np.nan)


@st.cache_data(show_spinner=False)
def load_league_map(db_path_str):
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
    league_map = load_league_map(str(DB_FILE))
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

        p_o15 = pair_prob(match.get("odds_ft_over15"), match.get("odds_ft_under15"))
        p_o25 = pair_prob(match.get("odds_ft_over25"), match.get("odds_ft_under25"))
        if not (np.isfinite(p_o15) and np.isfinite(p_o25)):
            continue
        if p_o15 < O15_MIN or p_o25 < O25_MIN:
            continue

        p_o05_2h = over05_second_half_if_ht00(match)
        rows.append(
            {
                "Campionato": lg,
                "Partita": f"{team_name(match, 'home')} - {team_name(match, 'away')}",
                "O1.5 pre": pct_text(p_o15),
                "O2.5 pre": pct_text(p_o25),
                "O0.5 2T se HT 0-0": pct_text(p_o05_2h),
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
        '<div class="goal2t-sub">Filtro fisso: O1.5 pre ≥ 80% · O2.5 pre ≥ 60% · O0.5 2T calcolato ipotizzando 0-0 all’intervallo</div>',
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
    st.caption(f"Partite ricevute da FootyStats per la data: {total_fixtures}. Sono mostrate solo le competizioni concordate e le partite che superano entrambi i filtri pre-match.")

    if not candidates:
        st.info("Nessuna partita della giornata rispetta contemporaneamente O1.5 ≥80% e O2.5 ≥60%.")
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
                "O0.5 2T se HT 0-0": st.column_config.TextColumn(width="medium"),
            },
        )

    with st.expander("Competizioni incluse"):
        st.write(" · ".join(sorted(SELECTED_LEAGUES)))


if __name__ == "__main__":
    main()
