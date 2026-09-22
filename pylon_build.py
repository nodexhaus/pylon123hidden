"""
Pylon: NFL anytime TD model build pipeline.

Downloads nflverse data, trains the model out-of-sample, prices the upcoming week,
and writes a self-contained site/index.html.

Usage:  python pylon_build.py                 # auto-detect season and week
        python pylon_build.py --season 2026 --week 4
"""
import argparse, datetime as dt, itertools, json, os, sys, urllib.request
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
REL = "https://github.com/nflverse/nflverse-data/releases/download/"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
os.makedirs(DATA, exist_ok=True)


# ----------------------------------------------------------------------------- download
def fetch(url, name, refresh):
    path = os.path.join(DATA, name)
    if os.path.exists(path) and not refresh:
        return path
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:
        print(f"  ! {name}: {e}")
        return None


def pq(tag, name, refresh=False):
    p = fetch(f"{REL}{tag}/{name}.parquet", name + ".parquet", refresh)
    return pd.read_parquet(p) if p else None


# ----------------------------------------------------------------------------- args
ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int)
ap.add_argument("--week", type=int)
ap.add_argument("--out", default=os.path.join(ROOT, "site", "index.html"))
args = ap.parse_args()

print("Downloading schedule and lines…")
games = pd.read_csv(fetch(GAMES_URL, "games.csv", True))
reg = games[games.game_type == "REG"]
if args.season:
    S = args.season
else:
    upcoming = reg[reg.result.isna()]
    S = int(upcoming.season.min()) if len(upcoming) else int(reg.season.max())
if args.week:
    W = args.week
else:
    left = reg[(reg.season == S) & reg.result.isna()]
    W = int(left.week.min()) if len(left) else int(reg[reg.season == S].week.max())
print(f"Building season {S}, week {W}")

TRAIN, TEST = S - 2, S - 1          # fit on TRAIN (prior TRAIN-1), score TEST blind
SEASONS = [S - 3, S - 2, S - 1, S]

# Current-season files refresh every run; past seasons are cached
pbp, snaps, part = {}, {}, {}
for y in SEASONS:
    fresh = y == S
    pbp[y] = pq("pbp", f"play_by_play_{y}", fresh)
    snaps[y] = pq("snap_counts", f"snap_counts_{y}", fresh)
    if y < S:
        part[y] = pq("pbp_participation", f"pbp_participation_{y}")
players_tbl = pq("players", "players", True)
rost = pq("weekly_rosters", f"roster_weekly_{S}", True)
inj = pq("injuries", f"injuries_{S}", True)
depth = pq("depth_charts", f"depth_charts_{S}", True)
teams_csv = pd.read_csv(fetch(REL + "teams/teams_colors_logos.csv", "teams.csv", False))

pfr2gsis = dict(zip(players_tbl.pfr_id, players_tbl.gsis_id))
POS = dict(zip(players_tbl.gsis_id, players_tbl.position))
SKILL = {"QB", "RB", "WR", "TE", "FB"}


# ----------------------------------------------------------------------------- prep
def prep(p):
    if p is None or len(p) == 0:
        return None
    p = p[(p.season_type == "REG") & p.posteam.notna()].copy()
    p["is_rush"] = (p.rush_attempt == 1) & (p.qb_kneel != 1) & p.rusher_player_id.notna()
    p["is_tgt"] = (p.pass_attempt == 1) & (p.sack != 1) & p.receiver_player_id.notna()
    p["rz"] = p.yardline_100 <= 20
    p["i10"] = p.yardline_100 <= 10
    p["ez"] = p.is_tgt & (p.air_yards >= p.yardline_100)
    return p


pbp = {y: prep(p) for y, p in pbp.items()}
EMPTY_PBP = pbp[S - 1].iloc[0:0]
for y in SEASONS:
    if pbp[y] is None:
        pbp[y] = EMPTY_PBP


def prep_snaps(s):
    if s is None or len(s) == 0:
        return pd.DataFrame(columns=["pid", "team", "week", "game_id", "pct", "snaps"])
    s = s[(s.game_type == "REG") & (s.offense_snaps > 0)].copy()
    s["pid"] = s.pfr_player_id.map(pfr2gsis)
    s = s[s.pid.notna()]
    return s.rename(columns={"offense_pct": "pct", "offense_snaps": "snaps"})[["pid", "team", "week", "game_id", "pct", "snaps"]]


snaps = {y: prep_snaps(s) for y, s in snaps.items()}

# nflverse uses LA for the Rams everywhere; PFR snap counts use LAR in some seasons
for y in snaps:
    snaps[y]["team"] = snaps[y].team.replace({"LAR": "LA", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA"})


def team_game_totals(p):
    if len(p) == 0:
        return pd.DataFrame(columns=["game_id", "posteam", "rush", "rz_rush", "i10_rush", "tgt", "rz_tgt", "ez_tgt"])
    q = p.assign(rzr=p.is_rush & p.rz, i10r=p.is_rush & p.i10, rzt=p.is_tgt & p.rz)
    return q.groupby(["game_id", "posteam"]).agg(rush=("is_rush", "sum"), rz_rush=("rzr", "sum"), i10_rush=("i10r", "sum"),
                                                 tgt=("is_tgt", "sum"), rz_tgt=("rzt", "sum"), ez_tgt=("ez", "sum")).reset_index()


def events(p):
    r = p[p.is_rush][["game_id", "week", "posteam", "rusher_player_id", "rz", "i10", "rush_touchdown", "yards_gained"]].rename(columns={"rusher_player_id": "pid"})
    t = p[p.is_tgt][["game_id", "week", "posteam", "receiver_player_id", "rz", "ez", "pass_touchdown", "yards_gained"]].rename(columns={"receiver_player_id": "pid"})
    return r, t


TG = {y: team_game_totals(pbp[y]) for y in SEASONS}
EV = {y: events(pbp[y]) for y in SEASONS}
METRICS = ["s_rush", "s_rz_rush", "s_i10_rush", "s_tgt", "s_rz_tgt", "s_ez_tgt"]


def shares(y, max_week=None):
    r, t = EV[y]
    tg = TG[y]
    if max_week is not None:
        r, t = r[r.week < max_week], t[t.week < max_week]
        tg = tg[tg.game_id.isin(set(pbp[y][pbp[y].week < max_week].game_id))]
    pr = r.groupby(["pid", "posteam"]).agg(rush=("rz", "size"), rz_rush=("rz", "sum"), i10_rush=("i10", "sum"), rush_td=("rush_touchdown", "sum"), ryd=("yards_gained", "sum"))
    pt = t.groupby(["pid", "posteam"]).agg(tgt=("rz", "size"), rz_tgt=("rz", "sum"), ez_tgt=("ez", "sum"), rec_td=("pass_touchdown", "sum"), cyd=("yards_gained", "sum"))
    df = pr.join(pt, how="outer").fillna(0).reset_index()
    app = pd.concat([r[["pid", "posteam", "game_id"]], t[["pid", "posteam", "game_id"]]]).drop_duplicates()
    tot = app.merge(tg, on=["game_id", "posteam"]).groupby(["pid", "posteam"]).agg(
        g=("game_id", "nunique"), T_rush=("rush", "sum"), T_rz_rush=("rz_rush", "sum"), T_i10_rush=("i10_rush", "sum"),
        T_tgt=("tgt", "sum"), T_rz_tgt=("rz_tgt", "sum"), T_ez_tgt=("ez_tgt", "sum")).reset_index()
    df = df.merge(tot, on=["pid", "posteam"])
    for m in ["rush", "rz_rush", "i10_rush", "tgt", "rz_tgt", "ez_tgt"]:
        df["s_" + m] = np.where(df["T_" + m] > 0, df[m] / df["T_" + m].replace(0, 1), 0.0)
    return df.rename(columns={"posteam": "team"})


def snap_feats(y, max_week=None):
    s = snaps[y] if max_week is None else snaps[y][snaps[y].week < max_week]
    if len(s) == 0:
        return pd.DataFrame(columns=["pid", "team", "sn", "smean", "slast"])
    s = s.sort_values("week")
    return s.groupby(["pid", "team"]).agg(sn=("pct", "size"), smean=("pct", "mean"), slast=("pct", "last")).reset_index()


def rz_snap_share(y):
    """Share of a team's red zone offensive plays each player was on the field for (participation data)."""
    pa = part.get(y)
    if pa is None:
        return {}
    p = pbp[y][(pbp[y].play_type.isin(["run", "pass"])) & pbp[y].rz][["game_id", "play_id", "posteam"]]
    m = pa[["nflverse_game_id", "play_id", "offense_players"]].rename(columns={"nflverse_game_id": "game_id"}).merge(p, on=["game_id", "play_id"])
    m = m[m.offense_players.notna() & (m.offense_players != "")]
    m["pid"] = m.offense_players.str.split(";")
    ex = m.explode("pid")
    ex = ex[ex.pid.str.len() > 0]
    player_games = ex[["pid", "posteam", "game_id"]].drop_duplicates()
    team_rz = m.groupby(["game_id", "posteam"]).size().rename("T").reset_index()
    denom = player_games.merge(team_rz, on=["game_id", "posteam"]).groupby(["pid", "posteam"]).T.sum()
    num = ex.groupby(["pid", "posteam"]).size()
    sh = (num / denom).dropna()
    # keep each player's main team value
    out = {}
    for (pid, team), v in sh.items():
        if pid not in out or num[(pid, team)] > out[pid][1]:
            out[pid] = (float(v), int(num[(pid, team)]))
    return {k: v[0] for k, v in out.items()}


def team_td_split(y, max_week=None):
    p = pbp[y] if max_week is None else pbp[y][pbp[y].week < max_week]
    o = p.groupby("posteam").agg(rtd=("rush_touchdown", "sum"), ptd=("pass_touchdown", "sum"))
    d = p.groupby("defteam").agg(rtd=("rush_touchdown", "sum"), ptd=("pass_touchdown", "sum"))
    return o, d


# ----------------------------------------------------------------------------- league constants
def td_per_pt(years):
    num = den = 0.0
    for y in years:
        p = pbp[y]
        tds = p.groupby(["game_id", "posteam"]).apply(lambda d: d.rush_touchdown.sum() + d.pass_touchdown.sum(), include_groups=False)
        for g in reg[reg.season == y].itertuples():
            if pd.isna(g.spread_line) or pd.isna(g.result):
                continue
            for team, imp in ((g.home_team, g.total_line / 2 + g.spread_line / 2), (g.away_team, g.total_line / 2 - g.spread_line / 2)):
                if (g.game_id, team) in tds.index:
                    num += tds[(g.game_id, team)]; den += imp
    return num / den


A = td_per_pt([S - 2, S - 1])
lp = pd.concat([pbp[S - 2], pbp[S - 1]])
LG_R = float(lp.rush_touchdown.sum() / (lp.rush_touchdown.sum() + lp.pass_touchdown.sum()))
print(f"TDs per implied point {A:.4f}, league rush TD share {LG_R:.3f}")


def team_r(team, opp, oc, dc, op, dp, beta, m=10):
    def frac(a, b, t):
        r = a.rtd.get(t, 0) * 2 + b.rtd.get(t, 0)
        tot = r + a.ptd.get(t, 0) * 2 + b.ptd.get(t, 0)
        return (r + m * LG_R) / (tot + m)
    off, de = frac(oc, op, team), frac(dc, dp, opp)
    return float(np.clip(off + beta * (de - LG_R), 0.15, 0.7)), off, de


# ----------------------------------------------------------------------------- assemble rows
def assemble(cur, prior, sc_cur, sc_pri, rzp, team_of):
    cg = cur.set_index(["pid", "team"])
    pg = {k: v for k, v in prior.groupby("pid")}
    scc = sc_cur.set_index(["pid", "team"])
    scp = {k: v for k, v in sc_pri.groupby("pid")}
    rows = []
    for pid, team in team_of.items():
        row = {"pid": pid, "team": team}
        if (pid, team) in cg.index:
            c = cg.loc[(pid, team)]
            row.update(n=int(c.g), td=int(c.rush_td + c.rec_td), c_car=float(c.rush), c_ryd=float(c.ryd), c_tg=float(c.tgt), c_cyd=float(c.cyd), **{"c_" + m: float(c[m]) for m in METRICS})
        else:
            row.update(n=0, td=0, c_car=0.0, c_ryd=0.0, c_tg=0.0, c_cyd=0.0, **{"c_" + m: 0.0 for m in METRICS})
        if pid in pg:
            pp = pg[pid]; same = pp[pp.team == team]
            src = same.iloc[0] if len(same) else pp.sort_values("g").iloc[-1]
            row.update(pn=int(src.g), psame=bool(len(same)), ptd=int(src.rush_td + src.rec_td), p_car=float(src.rush), p_ryd=float(src.ryd), p_tg=float(src.tgt), p_cyd=float(src.cyd), **{"p_" + m: float(src[m]) for m in METRICS})
        else:
            row.update(pn=0, psame=False, ptd=0, p_car=0.0, p_ryd=0.0, p_tg=0.0, p_cyd=0.0, **{"p_" + m: 0.0 for m in METRICS})
        if (pid, team) in scc.index:
            s = scc.loc[(pid, team)]; row.update(sn=int(s.sn), smean=float(s.smean), slast=float(s.slast))
        else:
            row.update(sn=0, smean=0.0, slast=0.0)
        if pid in scp:
            sp = scp[pid]; same = sp[sp.team == team]
            src = same.iloc[0] if len(same) else sp.sort_values("sn").iloc[-1]
            row.update(spn=int(src.sn), sprior=float(src.smean))
        else:
            row.update(spn=0, sprior=0.0)
        row["rzp"] = rzp.get(pid, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def blend(df, K, nt):
    kp = np.where(df.pn > 0, np.where(df.psame, K, K * nt), 0.0)
    d = df.n + kp
    return {m: np.where(d > 0, (df.n * df["c_" + m] + kp * df["p_" + m]) / np.where(d > 0, d, 1), 0) for m in METRICS}


def raw_lambda(df, P):
    w = blend(df, P["K"], P["newteam"])
    rush = P["w_i10"] * w["s_i10_rush"] + P["w_rz"] * w["s_rz_rush"] + (1 - P["w_i10"] - P["w_rz"]) * w["s_rush"]
    rec = P["w_ez"] * w["s_ez_tgt"] + P["w_rzt"] * w["s_rz_tgt"] + (1 - P["w_ez"] - P["w_rzt"]) * w["s_tgt"]
    return np.maximum(df.teamTD * (df.r * rush + (1 - df.r) * rec), 1e-4)


FEATS = ["log_raw", "snap", "snap_trend", "rz_snap_prior", "no_rz_prior", "is_qb", "is_te", "is_wr"]


def design(df, P):
    K = P["K"]
    kp = np.where(df.spn > 0, K, 0.0)
    d = df.sn + kp
    snap = np.where(d > 0, (df.sn * df.smean + kp * df.sprior) / np.where(d > 0, d, 1), 0)
    trend = np.where(df.sn > 0, df.slast - snap, 0)
    has = df.rzp.notna()
    X = np.column_stack([
        np.ones(len(df)), np.log(raw_lambda(df, P)), snap, trend,
        np.where(has, df.rzp.fillna(0), 0), (~has).astype(float),
        (df.pos == "QB").astype(float), (df.pos == "TE").astype(float), (df.pos == "WR").astype(float)])
    return X


def fit_cloglog(X, y, ridge=1.0, iters=40):
    """Binomial GLM with complementary log-log link: P = 1 - exp(-exp(Xb)). Ridge on non-intercept terms."""
    b = np.zeros(X.shape[1]); b[0] = np.log(-np.log(1 - y.mean()))
    R = np.eye(X.shape[1]) * ridge; R[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(X @ b, -8, 3)
        e = np.exp(eta); mu = np.clip(1 - np.exp(-e), 1e-6, 1 - 1e-6)
        dmu = e * np.exp(-e)
        w = dmu ** 2 / (mu * (1 - mu))
        z = eta + (y - mu) / np.maximum(dmu, 1e-9)
        b_new = np.linalg.solve(X.T @ (w[:, None] * X) + R, X.T @ (w * z) + R @ np.zeros_like(b))
        if np.max(np.abs(b_new - b)) < 1e-7:
            b = b_new; break
        b = b_new
    return b


def predict_glm(X, b):
    return 1 - np.exp(-np.exp(np.clip(X @ b, -8, 3)))


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ----------------------------------------------------------------------------- walk-forward season frames
def tinfo_for(T, wk):
    t = {}
    for g in reg[(reg.season == T) & (reg.week == wk)].itertuples():
        if pd.isna(g.spread_line):
            continue
        t[g.home_team] = (g.total_line / 2 + g.spread_line / 2, g.away_team, g.spread_line, g.total_line)
        t[g.away_team] = (g.total_line / 2 - g.spread_line / 2, g.home_team, -g.spread_line, g.total_line)
    return t


def week_frame(T, wk, prior, op, dp, sc_pri, rzp, sn, tdev, require_history=True):
    cur, (oc, dc), sc_cur = shares(T, wk), team_td_split(T, wk), snap_feats(T, wk)
    played = sn[sn.week == wk]
    df = assemble(cur, prior, sc_cur, sc_pri, rzp, dict(zip(played.pid, played.team)))
    if not len(df):
        return None
    if require_history:
        df = df[(df.n > 0) | (df.sn > 0)]
    tinfo = tinfo_for(T, wk)
    df = df[df.team.isin(tinfo)].copy()
    df["teamTD"] = A * df.team.map(lambda t: tinfo[t][0])
    df["opp"] = df.team.map(lambda t: tinfo[t][1])
    df["margin"] = df.team.map(lambda t: tinfo[t][2]); df["total"] = df.team.map(lambda t: tinfo[t][3])
    wr = EV[T][0]; wr = wr[wr.week == wk].groupby("pid").yards_gained.sum()
    df["ryd_act"] = df.pid.map(wr).fillna(0.0)
    wt = EV[T][1]; wt = wt[wt.week == wk].groupby("pid").yards_gained.sum()
    df["cyd_act"] = df.pid.map(wt).fillna(0.0)
    for b_ in (0.0, 0.25, 0.5):
        df[f"r_{b_}"] = [team_r(t, o, oc, dc, op, dp, b_)[0] for t, o in zip(df.team, df.opp)]
    df["pos"] = df.pid.map(POS)
    df["y"] = df.pid.isin(set(tdev[tdev.week == wk].td_player_id)).astype(int)
    df["week"] = wk
    df["naive"] = ((df.td + 0.5 * df.ptd * np.minimum(1, 8 / np.maximum(df.pn, 1)))
                   / (df.n + 0.5 * np.minimum(df.pn, 8)).clip(lower=1)).clip(0.01, 0.95)
    return df


def season_ctx(T):
    prior, (op, dp) = shares(T - 1), team_td_split(T - 1)
    sc_pri, rzp = snap_feats(T - 1), rz_snap_share(T - 1)
    tdev = pbp[T][pbp[T].td_player_id.notna() & ((pbp[T].rush_touchdown == 1) | (pbp[T].pass_touchdown == 1))]
    sn = snaps[T]; sn = sn[sn.pid.map(POS).isin(SKILL)]
    return prior, op, dp, sc_pri, rzp, sn, tdev


def season_frames(T):
    """Rows for every skill player who played an offensive snap in weeks 3-18 of season T."""
    ctx = season_ctx(T)
    out = [f for wk in range(3, int(pbp[T].week.max()) + 1) if (f := week_frame(T, wk, *ctx)) is not None]
    return pd.concat(out, ignore_index=True)


print("Building walk-forward frames…")
tr, te = season_frames(TRAIN), season_frames(TEST)
print(f"  train {TRAIN}: {len(tr)} player-games | test {TEST}: {len(te)} player-games")


def search_structural(d):
    best = None
    grid = itertools.product([1, 2, 3], [0.5], [0.3, 0.45, 0.6], [0.1, 0.2], [0.1, 0.25], [0.1, 0.2], [0.0, 0.25])
    for K, nt, wi, wr, we, wt, b in grid:
        P = dict(K=K, newteam=nt, w_i10=wi, w_rz=wr, w_ez=we, w_rzt=wt, beta=b)
        x = d.assign(r=d[f"r_{b}"])
        X = np.column_stack([np.ones(len(x)), np.log(raw_lambda(x, P))])
        bb = fit_cloglog(X, x.y.values, ridge=0.0, iters=25)
        s = ll(predict_glm(X, bb), x.y.values)
        if best is None or s < best[0]:
            best = (s, P)
    return best[1]


print("Fitting on", TRAIN)
P = search_structural(tr)
print("  structural", P)
trX = design(tr.assign(r=tr[f"r_{P['beta']}"]), P)
teX = design(te.assign(r=te[f"r_{P['beta']}"]), P)
b_full = fit_cloglog(trX, tr.y.values)
b_v1 = fit_cloglog(trX[:, :2], tr.y.values, ridge=0.0)
p_full, p_v1 = predict_glm(teX, b_full), predict_glm(teX[:, :2], b_v1)
yt = te.y.values
base = np.full(len(te), tr.y.mean())


def calib_bins(p, y):
    edges = [0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, 1]
    d = pd.DataFrame({"p": p, "y": y}); d["bin"] = pd.cut(d.p, edges)
    c = d.groupby("bin", observed=True).agg(pred=("p", "mean"), act=("y", "mean"), n=("y", "size")).reset_index()
    return [dict(pred=round(r.pred, 4), act=round(r.act, 4), n=int(r.n)) for r in c.itertuples()]


tt = te.assign(p=p_full)
top = tt.sort_values("p", ascending=False).groupby("week").head(20)
backtest = dict(
    train=TRAIN, test=TEST, n=int(len(te)), tds=int(yt.sum()), weeks=f"3-{int(te.week.max())}",
    logloss=ll(p_full, yt), logloss_v1=ll(p_v1, yt), logloss_naive=ll(te.naive.values, yt), logloss_base=ll(base, yt),
    brier=float(np.mean((p_full - yt) ** 2)), brier_naive=float(np.mean((te.naive.values - yt) ** 2)),
    calibration=calib_bins(p_full, yt), top20_pred=float(top.p.mean()), top20_act=float(top.y.mean()),
    by_pos={pos: dict(pred=float(g.p.mean()), act=float(g.y.mean()), n=int(len(g))) for pos, g in tt.groupby("pos") if pos in SKILL and len(g) > 50},
)
print(json.dumps({k: v for k, v in backtest.items() if k not in ("calibration", "by_pos")}, indent=1))

# Final coefficients for live use: refit on both seasons with the chosen structure
allX = np.vstack([trX, teX]); ally = np.concatenate([tr.y.values, yt])
b_live = fit_cloglog(allX, ally)
coef = dict(zip(["intercept"] + FEATS, [round(float(v), 5) for v in b_live]))
print("Live coefficients", coef)

# ----------------------------------------------------------------------------- props: yards and TD markets
def passers(y, max_week=None):
    p = pbp[y] if max_week is None else pbp[y][pbp[y].week < max_week]
    q = p[(p.pass_attempt == 1) & (p.sack != 1) & p.passer_player_id.notna()]
    return q.groupby(["passer_player_id", "posteam"]).agg(att=("play_id", "size"), pyd=("passing_yards", "sum"), ptd=("pass_touchdown", "sum"),
                                                          g=("game_id", "nunique")).reset_index().rename(columns={"passer_player_id": "pid", "posteam": "team"})


def pace(y, max_week=None):
    p = pbp[y] if max_week is None else pbp[y][pbp[y].week < max_week]
    pa = (p.pass_attempt == 1) & (p.sack != 1)
    o = p.assign(pa=pa).groupby("posteam").agg(ra=("is_rush", "sum"), pa=("pa", "sum"), g=("game_id", "nunique"))
    d = p.assign(pa=pa, ry=np.where(p.is_rush, p.yards_gained, 0), py=np.where(pa, p.passing_yards.fillna(0), 0)).groupby("defteam").agg(
        ra=("is_rush", "sum"), ry=("ry", "sum"), pa=("pa", "sum"), py=("py", "sum"))
    return o, d


def blend_pace(T, wk):
    oc, dc = pace(T, wk); op, dp = pace(T - 1)
    out = {}
    for t in set(op.index) | set(oc.index):
        def g(df, col): return float(df[col].get(t, 0)) if len(df) else 0.0
        gc, gp = g(oc, "g"), g(op, "g")
        den = 2 * gc + gp or 1
        rush_pg = (2 * g(oc, "ra") + g(op, "ra")) / den
        pass_pg = (2 * g(oc, "pa") + g(op, "pa")) / den
        ra, ry = 2 * g(dc, "ra") + g(dp, "ra"), 2 * g(dc, "ry") + g(dp, "ry")
        pa_, py = 2 * g(dc, "pa") + g(dp, "pa"), 2 * g(dc, "py") + g(dp, "py")
        out[t] = dict(rush_pg=rush_pg, pass_pg=pass_pg, def_ypc=(ry + 150 * LG_YPC) / (ra + 150), def_ypa=(py + 250 * LG_YPA) / (pa_ + 250))
    return out


lg = pd.concat([pbp[S - 2], pbp[S - 1]])
LG_YPC = float(lg[lg.is_rush].yards_gained.mean())
_pa = lg[(lg.pass_attempt == 1) & (lg.sack != 1)]
LG_YPA = float(_pa.passing_yards.fillna(0).mean())
POS_YPC = {pos: float(lg[lg.is_rush & (lg.rusher_player_id.map(POS) == pos)].yards_gained.mean()) for pos in ["QB", "RB", "WR", "TE"]}
_tg = lg[lg.is_tgt]
POS_YPT = {pos: float(_tg[_tg.receiver_player_id.map(POS) == pos].yards_gained.mean()) for pos in ["RB", "WR", "TE", "QB"]}
YPT_PRIOR = 40.0
print("Yards per target by pos", POS_YPT)
print(f"League YPC {LG_YPC:.2f}, YPA {LG_YPA:.2f}, by pos {POS_YPC}")
YPC_PRIOR, YPA_PRIOR = 60.0, 200.0


def ypc_blend(c_car, c_ryd, p_car, p_ryd, pos):
    base = POS_YPC.get(pos, LG_YPC)
    return (c_ryd + 0.7 * p_ryd + YPC_PRIOR * base) / (c_car + 0.7 * p_car + YPC_PRIOR)


def rush_frames(T):
    ctx = season_ctx(T); rows = []
    for wk in range(3, int(pbp[T].week.max()) + 1):
        f = week_frame(T, wk, *ctx)
        if f is None: continue
        pc = blend_pace(T, wk)
        f = f[f.pos.isin(["QB", "RB", "WR"])].copy()
        f["hist_rush"] = f.team.map(lambda t: pc.get(t, {}).get("rush_pg", 26))
        f["dypc"] = f.opp.map(lambda t: pc.get(t, {}).get("def_ypc", LG_YPC))
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def rec_frames(T):
    ctx = season_ctx(T); rows = []
    for wk in range(3, int(pbp[T].week.max()) + 1):
        f = week_frame(T, wk, *ctx)
        if f is None: continue
        pc = blend_pace(T, wk)
        f = f[f.pos.isin(["WR", "TE", "RB"])].copy()
        f["hist_pass"] = f.team.map(lambda t: pc.get(t, {}).get("pass_pg", 33))
        f["dypa"] = f.opp.map(lambda t: pc.get(t, {}).get("def_ypa", LG_YPA))
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def team_frames(T):
    rows = []
    for wk in range(3, int(pbp[T].week.max()) + 1):
        pc = blend_pace(T, wk); ti = tinfo_for(T, wk)
        cur = pbp[T][pbp[T].week == wk]
        pa = (cur.pass_attempt == 1) & (cur.sack != 1)
        act = cur.assign(pa=pa).groupby("posteam").agg(ra=("is_rush", "sum"), pa=("pa", "sum"))
        for t, (imp, opp, mg, tot) in ti.items():
            if t in act.index and t in pc:
                rows.append(dict(team=t, week=wk, hist_rush=pc[t]["rush_pg"], hist_pass=pc[t]["pass_pg"], margin=mg, total=tot,
                                 ra=act.ra[t], pa=act.pa[t]))
    return pd.DataFrame(rows)


def qb_frames(T):
    rows = []
    for wk in range(3, int(pbp[T].week.max()) + 1):
        pc = blend_pace(T, wk); ti = tinfo_for(T, wk)
        cur, pri = passers(T, wk), passers(T - 1)
        (oc, dc), (op, dp) = team_td_split(T, wk), team_td_split(T - 1)
        wkp = passers(T, wk + 1); wkp = wkp.merge(passers(T, wk), on=["pid", "team"], how="left", suffixes=("", "_b")).fillna(0)
        wkp["att_w"] = wkp.att - wkp.att_b; wkp["pyd_w"] = wkp.pyd - wkp.pyd_b; wkp["ptd_w"] = wkp.ptd - wkp.ptd_b
        starters = wkp[wkp.att_w > 0].sort_values("att_w").groupby("team").tail(1)
        for s in starters.itertuples():
            if s.team not in ti or s.team not in pc: continue
            imp, opp, mg, tot = ti[s.team]
            c = cur[(cur.pid == s.pid)]; pp = pri[pri.pid == s.pid]
            rows.append(dict(pid=s.pid, team=s.team, week=wk, att_c=c.att.sum(), pyd_c=c.pyd.sum(), att_p=pp.att.sum(), pyd_p=pp.pyd.sum(),
                             hist_pass=pc[s.team]["pass_pg"], hist_rush=pc[s.team]["rush_pg"], dypa=pc.get(opp, {}).get("def_ypa", LG_YPA),
                             margin=mg, total=tot, impl=imp, r=team_r(s.team, opp, oc, dc, op, dp, P["beta"])[0],
                             pyd=s.pyd_w, ptd=s.ptd_w))
    return pd.DataFrame(rows)


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X]); return np.linalg.lstsq(X1, y, rcond=None)[0]


print("Fitting prop models on", TRAIN)
tf_tr, tf_te = team_frames(TRAIN), team_frames(TEST)
b_ra = ols(tf_tr[["hist_rush", "margin", "total"]].values, tf_tr.ra.values)
b_pa = ols(tf_tr[["hist_pass", "margin", "total"]].values, tf_tr.pa.values)


def team_ra(h, m, t): return b_ra[0] + b_ra[1] * h + b_ra[2] * m + b_ra[3] * t
def team_pa(h, m, t): return b_pa[0] + b_pa[1] * h + b_pa[2] * m + b_pa[3] * t


def rush_base(f):
    sh = blend(f, P["K"], P["newteam"])["s_rush"]
    ypc = np.array([ypc_blend(a, b_, c, d, pos) for a, b_, c, d, pos in zip(f.c_car, f.c_ryd, f.p_car, f.p_ryd, f.pos)])
    return team_ra(f.hist_rush, f.margin, f.total) * sh * ypc * np.sqrt(f.dypc / LG_YPC)


def ypa_blend(att_c, pyd_c, att_p, pyd_p):
    return (pyd_c + 0.7 * pyd_p + YPA_PRIOR * LG_YPA) / (att_c + 0.7 * att_p + YPA_PRIOR)


def pass_base(q):
    return team_pa(q.hist_pass, q.margin, q.total) * ypa_blend(q.att_c, q.pyd_c, q.att_p, q.pyd_p) * np.sqrt(q.dypa / LG_YPA)


rf_tr, rf_te = rush_frames(TRAIN), rush_frames(TEST)
cf_tr, cf_te = rec_frames(TRAIN), rec_frames(TEST)


def rec_base(f):
    sh = blend(f, P["K"], P["newteam"])["s_tgt"]
    ypt = np.array([(cy + .7 * py + YPT_PRIOR * POS_YPT.get(pos, 7.5)) / (ct + .7 * pt + YPT_PRIOR)
                    for ct, cy, pt, py, pos in zip(f.c_tg, f.c_cyd, f.p_tg, f.p_cyd, f.pos)])
    return team_pa(f.hist_pass, f.margin, f.total) * sh * ypt * np.sqrt(f.dypa / LG_YPA)
qf_tr, qf_te = qb_frames(TRAIN), qb_frames(TEST)
b_ry = ols(rush_base(rf_tr).values[:, None], rf_tr.ryd_act.values)
b_py = ols(pass_base(qf_tr).values[:, None], qf_tr.pyd.values)
b_cy = ols(rec_base(cf_tr).values[:, None], cf_tr.cyd_act.values)
pc_tr = b_cy[0] + b_cy[1] * rec_base(cf_tr).values


def fit_sd(pred, y):
    res = np.abs(y - pred) * np.sqrt(np.pi / 2)
    return ols(pred[:, None], res)


pr_tr = b_ry[0] + b_ry[1] * rush_base(rf_tr).values
pp_tr = b_py[0] + b_py[1] * pass_base(qf_tr).values
sd_ry, sd_py = fit_sd(pr_tr, rf_tr.ryd_act.values), fit_sd(pp_tr, qf_tr.pyd.values)
sd_cy = fit_sd(pc_tr, cf_tr.cyd_act.values)
med_ry = float(np.median(rf_tr.ryd_act.values - pr_tr)); med_py = float(np.median(qf_tr.pyd.values - pp_tr)); med_cy = float(np.median(cf_tr.cyd_act.values - pc_tr))
print("median offsets", med_ry, med_py, med_cy)
lam_ptd_tr = A * qf_tr.impl * (1 - qf_tr.r)
k_ptd = float(qf_tr.ptd.sum() / lam_ptd_tr.sum())

# blind 2025 evaluation
from math import erf
def pover(line, mu, sd): return 0.5 * (1 - np.vectorize(erf)((line - mu) / (np.maximum(sd, 1) * np.sqrt(2))))

pr_te = b_ry[0] + b_ry[1] * rush_base(rf_te).values
pp_te = b_py[0] + b_py[1] * pass_base(qf_te).values
naive_r = ((rf_te.c_ryd + 0.5 * rf_te.p_ryd * np.minimum(1, 8 / np.maximum(rf_te.pn, 1))) / (rf_te.n + 0.5 * np.minimum(rf_te.pn, 8)).clip(lower=1)).values
gc = np.where(qf_te.att_c > 0, np.maximum(1, np.round(qf_te.att_c / 30)), 0)
prior_pg = np.where(qf_te.att_p > 0, qf_te.pyd_p / np.maximum(1, np.round(qf_te.att_p / 30)), LG_YPA * 33)
naive_p = ((qf_te.pyd_c + 4 * prior_pg) / (gc + 4)).values
lam_te = k_ptd * A * qf_te.impl * (1 - qf_te.r)
ptd_ladder = [dict(line=l, pred=float((1 - sum(np.exp(-lam_te) * lam_te ** k / __import__('math').factorial(k) for k in range(int(l + .5)))).mean()),
                   act=float((qf_te.ptd >= l + .5).mean())) for l in (0.5, 1.5, 2.5)]


def cal_over(pred, sd, y, med=0.0):
    """Lines set around the model's median: tests both the center and the spread."""
    out = []
    for off in (-20, -10, 0, 10, 20):
        line = pred + med + off
        p = pover(line, pred + med, sd)
        out.append(dict(off=off, pred=float(p.mean()), act=float((y > line).mean())))
    return out


sd_r_te = sd_ry[0] + sd_ry[1] * pr_te; sd_p_te = sd_py[0] + sd_py[1] * pp_te
pc_te = b_cy[0] + b_cy[1] * rec_base(cf_te).values
sd_c_te = sd_cy[0] + sd_cy[1] * pc_te
naive_c = ((cf_te.c_cyd + 0.5 * cf_te.p_cyd * np.minimum(1, 8 / np.maximum(cf_te.pn, 1))) / (cf_te.n + 0.5 * np.minimum(cf_te.pn, 8)).clip(lower=1)).values
props_bt = dict(
    ryd=dict(n=int(len(rf_te)), mae=float(np.mean(np.abs(pr_te - rf_te.ryd_act))), mae_naive=float(np.mean(np.abs(naive_r - rf_te.ryd_act))),
             mae_flat=float(np.mean(np.abs(rf_tr.ryd_act.mean() - rf_te.ryd_act))),
             corr=float(np.corrcoef(pr_te, rf_te.ryd_act)[0, 1]), cal=cal_over(pr_te, sd_r_te, rf_te.ryd_act.values, med_ry)),
    pyd=dict(n=int(len(qf_te)), mae=float(np.mean(np.abs(pp_te - qf_te.pyd))), mae_naive=float(np.mean(np.abs(naive_p - qf_te.pyd))),
             mae_flat=float(np.mean(np.abs(qf_tr.pyd.mean() - qf_te.pyd))),
             corr=float(np.corrcoef(pp_te, qf_te.pyd)[0, 1]), cal=cal_over(pp_te, sd_p_te, qf_te.pyd.values, med_py)),
    cyd=dict(n=int(len(cf_te)), mae=float(np.mean(np.abs(pc_te - cf_te.cyd_act))), mae_naive=float(np.mean(np.abs(naive_c - cf_te.cyd_act))),
             mae_flat=float(np.mean(np.abs(cf_tr.cyd_act.mean() - cf_te.cyd_act))),
             corr=float(np.corrcoef(pc_te, cf_te.cyd_act)[0, 1]), cal=cal_over(pc_te, sd_c_te, cf_te.cyd_act.values, med_cy)),
    ptd=dict(n=int(len(qf_te)), ladder=ptd_ladder, mean_pred=float(lam_te.mean()), mean_act=float(qf_te.ptd.mean())),
)
def rush_share(f):
    w = blend(f, P["K"], P["newteam"])
    rush = P["w_i10"] * w["s_i10_rush"] + P["w_rz"] * w["s_rz_rush"] + (1 - P["w_i10"] - P["w_rz"]) * w["s_rush"]
    rec = P["w_ez"] * w["s_ez_tgt"] + P["w_rzt"] * w["s_rz_tgt"] + (1 - P["w_ez"] - P["w_rzt"]) * w["s_tgt"]
    a_, b2 = f.r * rush, (1 - f.r) * rec
    return np.where(a_ + b2 > 0, a_ / np.maximum(a_ + b2, 1e-9), 0)


te_r = te.assign(r=te[f"r_{P['beta']}"])
lam_any = -np.log(1 - np.clip(p_full, 1e-6, 1 - 1e-6))
p_rtd = 1 - np.exp(-lam_any * rush_share(te_r))
rtd_ev = pbp[TEST][(pbp[TEST].rush_touchdown == 1) & pbp[TEST].td_player_id.notna()][["week", "td_player_id"]]
rset = set(zip(rtd_ev.week, rtd_ev.td_player_id))
y_rtd = np.array([(w, pid) in rset for w, pid in zip(te.week, te.pid)]).astype(int)
props_bt["rtd"] = dict(n=int(len(te)), tds=int(y_rtd.sum()), logloss=ll(p_rtd, y_rtd), logloss_base=ll(np.full(len(te), y_rtd.mean()), y_rtd),
                       calibration=calib_bins(p_rtd, y_rtd))
print("rush TD", {k: v for k, v in props_bt["rtd"].items() if k != "calibration"}, props_bt["rtd"]["calibration"])
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("cal", "ladder", "calibration")} for k, v in props_bt.items()}, indent=1))
print("ptd ladder", ptd_ladder)
print("ryd cal", props_bt["ryd"]["cal"]); print("pyd cal", props_bt["pyd"]["cal"])

# refit on both seasons for live use
tf_all = pd.concat([tf_tr, tf_te]); rf_all = pd.concat([rf_tr, rf_te]); cf_all = pd.concat([cf_tr, cf_te]); qf_all = pd.concat([qf_tr, qf_te])
b_ra = ols(tf_all[["hist_rush", "margin", "total"]].values, tf_all.ra.values)
b_pa = ols(tf_all[["hist_pass", "margin", "total"]].values, tf_all.pa.values)
b_ry = ols(rush_base(rf_all).values[:, None], rf_all.ryd_act.values)
b_py = ols(pass_base(qf_all).values[:, None], qf_all.pyd.values)
sd_ry = fit_sd(b_ry[0] + b_ry[1] * rush_base(rf_all).values, rf_all.ryd_act.values)
sd_py = fit_sd(b_py[0] + b_py[1] * pass_base(qf_all).values, qf_all.pyd.values)
k_ptd = float(qf_all.ptd.sum() / (A * qf_all.impl * (1 - qf_all.r)).sum())
b_cy = ols(rec_base(cf_all).values[:, None], cf_all.cyd_act.values)
sd_cy = fit_sd(b_cy[0] + b_cy[1] * rec_base(cf_all).values, cf_all.cyd_act.values)
med_ry = float(np.median(rf_all.ryd_act.values - (b_ry[0] + b_ry[1] * rush_base(rf_all).values)))
med_py = float(np.median(qf_all.pyd.values - (b_py[0] + b_py[1] * pass_base(qf_all).values)))
med_cy = float(np.median(cf_all.cyd_act.values - (b_cy[0] + b_cy[1] * rec_base(cf_all).values)))
PROPS = dict(b_ra=[round(float(x), 4) for x in b_ra], b_pa=[round(float(x), 4) for x in b_pa], b_ry=[round(float(x), 4) for x in b_ry],
             b_py=[round(float(x), 4) for x in b_py], sd_ry=[round(float(x), 4) for x in sd_ry], sd_py=[round(float(x), 4) for x in sd_py],
             b_cy=[round(float(x), 4) for x in b_cy], sd_cy=[round(float(x), 4) for x in sd_cy],
             POS_YPT={k: round(v, 3) for k, v in POS_YPT.items()}, YPT_PRIOR=YPT_PRIOR,
             med_ry=round(med_ry, 3), med_py=round(med_py, 3), med_cy=round(med_cy, 3),
             k_ptd=round(k_ptd, 4), LG_YPC=round(LG_YPC, 3), LG_YPA=round(LG_YPA, 3), POS_YPC={k: round(v, 3) for k, v in POS_YPC.items()},
             YPC_PRIOR=YPC_PRIOR, YPA_PRIOR=YPA_PRIOR, backtest=props_bt)
print("Live prop coefficients", {k: v for k, v in PROPS.items() if k != "backtest"})


# ----------------------------------------------------------------------------- this season's replay
# Price each completed week of the current season with only the data available before it, then attach results.
history = []
if len(pbp[S]):
    ctx = season_ctx(S)
    for wk in sorted(int(w) for w in pbp[S].week.unique()):
        f = week_frame(S, wk, *ctx, require_history=False)
        if f is None or not len(f):
            continue
        f = f.assign(r=f[f"r_{P['beta']}"])
        f["p"] = predict_glm(design(f, P), b_live)
        f = f[f.p >= 0.04].sort_values("p", ascending=False)
        names = dict(zip(players_tbl.gsis_id, players_tbl.display_name))
        top = f.head(20)
        history.append(dict(
            week=wk, n=int(len(f)), tds=int(f.y.sum()), exp=round(float(f.p.sum()), 1),
            top20_exp=round(float(top.p.sum()), 1), top20_hit=int(top.y.sum()),
            rows=[[names.get(r.pid, r.pid), r.pos, r.team, r.opp, int(round(r.p * 1000)), int(r.y)] for r in f.itertuples()]))
    print("Replayed weeks:", [(h["week"], f"{h['top20_hit']}/20 vs {h['top20_exp']} exp") for h in history])


# ----------------------------------------------------------------------------- live slate
print("Pricing the slate…")
cur, prior = shares(S), shares(S - 1)
oc, dc = team_td_split(S); op, dp = team_td_split(S - 1)
sc_cur, sc_pri, rzp = snap_feats(S), snap_feats(S - 1), rz_snap_share(S - 1)

ro = rost[rost.gsis_id.notna()].sort_values("week").groupby("gsis_id").tail(1)
ro = ro[ro.position.isin(SKILL) & (ro.status == "ACT")]
live = assemble(cur, prior, sc_cur, sc_pri, rzp, dict(zip(ro.gsis_id, ro.team)))
live = live[(live.n > 0) | (live.pn > 0) | (live.sn > 0) | (live.spn > 0)]
meta = ro.set_index("gsis_id")
live["name"] = live.pid.map(meta.full_name)
live["pos"] = live.pid.map(meta.position)
live["num"] = live.pid.map(meta.jersey_number).fillna(0).astype(int)
live["rookie"] = live.pid.map(meta.years_exp) == 0

# depth chart: latest snapshot, best rank at the player's own position
depth_rank = {}
if depth is not None and len(depth):
    dl = depth[depth.dt == depth.dt.max()]
    dl = dl[dl.pos_abb.isin(["QB", "RB", "WR", "TE", "FB"])]
    depth_rank = dl.groupby("gsis_id").pos_rank.min().to_dict()
    depth_stamp = str(depth.dt.max())[:10]
else:
    depth_stamp = None
live["depth"] = live.pid.map(depth_rank)

injmap, inj_week = {}, None
if inj is not None and len(inj):
    inj_week = int(inj.week.max())
    li = inj[inj.week == inj_week]
    injmap = {r.gsis_id: [r.report_status if isinstance(r.report_status, str) else None,
                          r.report_primary_injury if isinstance(r.report_primary_injury, str) else None] for r in li.itertuples()}

gm = reg[(reg.season == S) & (reg.week == W)]
SD = 13.5                                      # std. dev. of NFL margins and totals around the line
PDF0 = 1 / (SD * np.sqrt(2 * np.pi))


def no_vig(a, b):
    """Two American prices -> vig-free probability of side a (None if missing)."""
    if pd.isna(a) or pd.isna(b):
        return None
    pa = 100 / (a + 100) if a > 0 else -a / (-a + 100)
    pb = 100 / (b + 100) if b > 0 else -b / (-b + 100)
    return pa / (pa + pb)


def fz(x):
    return None if pd.isna(x) else x


slate, tinfo = [], {}
for g in gm.itertuples():
    sp = 0.0 if pd.isna(g.spread_line) else float(g.spread_line)
    tot = 44.0 if pd.isna(g.total_line) else float(g.total_line)
    # Juice-adjust: shift each line by how far the vig-free price sits from 50/50
    p_over = no_vig(g.over_odds, g.under_odds)
    p_home = no_vig(g.home_spread_odds, g.away_spread_odds)
    tot_adj = tot + ((p_over - 0.5) / PDF0 if p_over is not None else 0)
    sp_adj = sp + ((p_home - 0.5) / PDF0 if p_home is not None else 0)
    slate.append(dict(id=g.game_id, day=g.gameday, time=g.gametime, weekday=g.weekday, home=g.home_team, away=g.away_team,
                      spread=sp, total=tot, spread_adj=round(sp_adj, 2), total_adj=round(tot_adj, 2),
                      over_odds=fz(g.over_odds), under_odds=fz(g.under_odds), hml=fz(g.home_moneyline), aml=fz(g.away_moneyline),
                      home_qb=fz(g.home_qb_id), away_qb=fz(g.away_qb_id), home_qb_name=fz(g.home_qb_name), away_qb_name=fz(g.away_qb_name),
                      temp=fz(g.temp), wind=fz(g.wind),
                      roof=g.roof if isinstance(g.roof, str) else None, noline=bool(pd.isna(g.spread_line))))
    tinfo[g.home_team] = g.away_team; tinfo[g.away_team] = g.home_team
live = live[live.team.isin(tinfo)]

# Unit strength: TDs per game scored/allowed, current season weighted 2x, as league percentiles
def per_game(key):
    frames = []
    for y, wgt in ((S - 1, 1.0), (S, 2.0)):
        p = pbp[y]
        if not len(p):
            continue
        g = p.groupby(key).agg(r=("rush_touchdown", "sum"), pa=("pass_touchdown", "sum"), gp=("game_id", "nunique"))
        frames.append(g * wgt)
    t = sum(frames)
    return pd.DataFrame({"rush": t.r / t.gp, "pass": t.pa / t.gp})


uo, ud = per_game("posteam"), per_game("defteam")
units = {}
for t in uo.index:
    units[t] = dict(rushO=round(float((uo["rush"] < uo.loc[t, "rush"]).mean()), 3),
                    passO=round(float((uo["pass"] < uo.loc[t, "pass"]).mean()), 3),
                    rushD=round(float((ud["rush"] > ud.loc[t, "rush"]).mean()), 3),   # higher = stingier
                    passD=round(float((ud["pass"] > ud.loc[t, "pass"]).mean()), 3))

teams = {}
for t, opp in tinfo.items():
    _, off, _ = team_r(t, opp, oc, dc, op, dp, 0)
    _, _, de = team_r(opp, t, oc, dc, op, dp, 0)
    teams[t] = dict(opp=opp, offR=round(off, 4), defR=round(de, 4), units=units.get(t))

r_, t_ = EV[S]
wl = (r_.groupby(["pid", "week"]).agg(car=("rz", "size"), rzc=("rz", "sum"), rtd=("rush_touchdown", "sum"))
      .join(t_.groupby(["pid", "week"]).agg(tgt=("rz", "size"), rzt=("rz", "sum"), ctd=("pass_touchdown", "sum")), how="outer").fillna(0).reset_index())
snapwk = snaps[S].set_index(["pid", "week"]).pct.to_dict()
logs = {}
for r in wl.itertuples():
    logs.setdefault(r.pid, []).append([int(r.week), int(r.car), int(r.tgt), int(r.rzc + r.rzt), int(r.rtd + r.ctd), round(snapwk.get((r.pid, r.week), 0), 2)])

tc = teams_csv.set_index("team_abbr")
colors = {t: [tc.loc[t, "team_color"], tc.loc[t, "team_color2"], tc.loc[t, "team_nick"]] for t in tinfo if t in tc.index}

pc_live = blend_pace(S, W)
qp_c, qp_p = passers(S), passers(S - 1)
qbstat = {}
for pid in live[live.pos == "QB"].pid:
    c = qp_c[qp_c.pid == pid]; p_ = qp_p[qp_p.pid == pid]
    qbstat[pid] = [int(c.att.sum()), int(c.pyd.sum()), int(p_.att.sum()), int(p_.pyd.sum())]
for t in teams:
    pcv = pc_live.get(t, {})
    teams[t].update(rush_pg=round(pcv.get("rush_pg", 26), 2), pass_pg=round(pcv.get("pass_pg", 33), 2),
                    def_ypc=round(pcv.get("def_ypc", LG_YPC), 3), def_ypa=round(pcv.get("def_ypa", LG_YPA), 3))

players = []
for r in live.itertuples():
    players.append(dict(
        id=r.pid, name=r.name, pos=r.pos, team=r.team, num=int(r.num), rookie=bool(r.rookie),
        n=int(r.n), td=int(r.td), pn=int(r.pn), psame=bool(r.psame), ptd=int(r.ptd),
        c=[round(getattr(r, "c_" + m), 4) for m in METRICS], p=[round(getattr(r, "p_" + m), 4) for m in METRICS],
        sn=int(r.sn), smean=round(r.smean, 3), slast=round(r.slast, 3), spn=int(r.spn), sprior=round(r.sprior, 3),
        rzp=None if pd.isna(r.rzp) else round(float(r.rzp), 3),
        ru=[int(r.c_car), int(r.c_ryd), int(r.p_car), int(r.p_ryd)], rc=[int(r.c_tg), int(r.c_cyd), int(r.p_tg), int(r.p_cyd)], qb=qbstat.get(r.pid),
        depth=None if pd.isna(r.depth) else int(r.depth),
        inj=injmap.get(r.pid), log=logs.get(r.pid, [])))

data = dict(season=S, week=W, built=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            through=int(pbp[S].week.max()) if len(pbp[S]) else 0, inj_week=inj_week, depth_date=depth_stamp,
            A=round(A, 5), LG_R=round(LG_R, 4), params=P, coef=coef, backtest=backtest, history=history, props=PROPS,
            games=slate, teams=teams, colors=colors, players=players)

tpl = open(os.path.join(ROOT, "template.html"), encoding="utf-8").read()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
payload = json.dumps(data, separators=(",", ":"), default=lambda o: None).replace("</", "<\\/")
open(args.out, "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/null", payload))
print(f"Wrote {args.out} ({len(players)} players, {len(slate)} games)")
