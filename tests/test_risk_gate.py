"""
Risk gate unit tests.
The gate is the most safety-critical component — test every path.
"""
import pytest
from services.risk_gate import RiskGate


@pytest.fixture
def gate():
    """Fresh RiskGate instance for each test."""
    return RiskGate()


def _check(gate, *, stop_loss=1.0800, spread_pips=1.2, age=0.5, source="agent"):
    """Helper: call approve_order with sensible defaults."""
    return gate.approve_order(
        pair="EUR/USD",
        direction="BUY",
        size=1000,
        entry_price=1.0850,
        stop_loss=stop_loss,
        spread_pips=spread_pips,
        data_age_seconds=age,
        source=source,
    )


def test_gate_open_by_default(gate):
    """Fresh gate with valid inputs must pass."""
    assert _check(gate).allowed


def test_kill_switch_blocks_all_trades(gate):
    """Kill switch must block any trade, no exceptions."""
    gate.activate_kill_switch("test")
    d = _check(gate)
    assert not d.allowed
    assert "kill" in d.reason.lower()


def test_kill_switch_deactivates(gate):
    gate.activate_kill_switch("test")
    gate.deactivate_kill_switch()
    assert _check(gate).allowed


def test_manual_only_blocks_agent_orders(gate):
    gate.set_manual_only(True)
    assert not _check(gate, source="agent").allowed


def test_manual_only_allows_human_orders(gate):
    gate.set_manual_only(True)
    assert _check(gate, source="human").allowed


def test_shadow_mode_passes_but_flags(gate):
    gate.set_shadow_mode(True)
    d = _check(gate)
    assert d.allowed
    assert d.shadow


def test_drawdown_breaker_activates_kill_switch(gate):
    gate._max_drawdown_pct = 10.0
    triggered = gate.check_drawdown(11.0)
    assert triggered
    assert gate.kill_switch_active


def test_drawdown_below_limit_does_not_trigger(gate):
    gate._max_drawdown_pct = 10.0
    assert not gate.check_drawdown(9.0)
    assert not gate.kill_switch_active


def test_stale_data_blocked(gate):
    d = _check(gate, age=90)
    assert not d.allowed
    assert "stale" in d.reason.lower()


def test_wide_spread_blocked(gate):
    assert not _check(gate, spread_pips=6.0).allowed


def test_missing_stop_loss_blocked(gate):
    assert not _check(gate, stop_loss=None).allowed
