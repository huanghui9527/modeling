from __future__ import annotations

from zrt.training.io.config_loader import _parse_strategy
from zrt.training.search.training_search_util import _make_strategy_from_config
from zrt.training.spec.strategy import Strategy


def test_strategy_defaults_keep_mega_moe_disabled():
    strategy = Strategy()

    assert strategy.mega_moe is False
    assert strategy.mega_moe_waves == 0


def test_parse_strategy_accepts_mega_moe_switch():
    strategy = _parse_strategy({
        "ep": 8,
        "mega_moe": True,
        "mega_moe_waves": 4,
    })

    assert strategy.ep == 8
    assert strategy.mega_moe is True
    assert strategy.mega_moe_waves == 4


def test_search_strategy_accepts_mega_moe_switch():
    strategy = _make_strategy_from_config({
        "ep": 8,
        "mega_moe": True,
        "mega_moe_waves": 2,
    })

    assert strategy.ep == 8
    assert strategy.mega_moe is True
    assert strategy.mega_moe_waves == 2
