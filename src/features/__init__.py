"""One module per angle. Each exposes `signal(ctx, t) -> pd.Series` indexed by permno, with the
pre-registered direction already applied (higher = predicted higher return). Only data dated
<= t may be used; tests/test_features.py enforces it by truncation."""
from importlib import import_module

ANGLES = ["P3_gibrat_up", "P4_tail_momentum", "P7_compounder", "P9_hype_max", "P9_hype_composite",
          "A4_insider_buys", "A5_lazy_filings", "A8_rd_underpriced", "A9_fluent_ticker",
          "B4_named_as_rival", "B6_crisis_share_gainer", "B7_second_derivative", "B8_scale_gets_cheaper",
          "B11_boring_dinner", "N1_quality_minus_junk", "N2_lindy_age", "N3_volume_spike", "N4_low_beta_tail_safe"]
# portfolio-construction angles handled in backtest.py: P2_zipf_switch, P5_early_entrants, P5_no_trim
# trained angles handled in backtest.py via their modules' fit(): P6_tail_classifier, P8_survival


def get(angle):
    return import_module(f"src.features.{angle.lower()}")
