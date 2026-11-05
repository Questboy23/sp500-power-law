"""python -m src.live.run init YYYY-MM-DD | daily"""
import sys, json, pandas as pd
from src.live import engine
from src.live.signals import compute

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "init":
        d = pd.Timestamp(sys.argv[2])
        w, asof = engine.index_weights(save=False)
        k, info = compute(w, d)
        (engine.LIVE / "shadow_signal_info.json").write_text(json.dumps(info, indent=1))
        s = engine.init(str(d.date()), k, note="SYSTEM = S&P 500 + rule I3; SHADOW = LIVE_SHADOW (failed gate)")
        print("initialized", s["start_date"], "missing:", s["missing_at_start"])
    elif cmd == "daily":
        t = engine.daily()
        print(len(t), "trades")
    print(open(engine.MARKS).read()[-600:])
