"""PP stitcher bubble 计算与约束验证测试。

验证原理：
- bubble = step_time - M * max(fwd+bwd) 应等于瓶颈设备的空闲时间（当 p2p=0 时）
- ZeroBubble 三约束：F→B_dx 激活、bwd_dx→bwd_dw 梯度链、bwd_dx 跨 stage P2P
- bwd_dw 无跨 stage P2P（权重梯度是本地计算，不跨 stage 传递）
- DualPipeV 的 _vstage_to_device 应映射到真实物理设备（stream_id ∈ {0..pp-1})
"""

import pytest
from python.zrt.executor.pp_stitcher import PPStitcher, GridTask


def _bottleneck_idle(result):
    """从调度结果中计算瓶颈设备空闲时间 = step_time - 该设备总计算量。"""
    busy_by_stream = {}
    for t in result.tasks:
        busy_by_stream[t.stream_id] = busy_by_stream.get(t.stream_id, 0.0) + t.latency_us
    idle_by_stream = {s: result.step_time_us - b for s, b in busy_by_stream.items()}
    return max(idle_by_stream.values()), idle_by_stream, busy_by_stream


class TestBubbleMatchesSchedule:
    """旧公式 bubble 应等于瓶颈设备空闲时间（p2p=0 时精确成立）。"""

    def test_1f1b_bubble_equals_bottleneck_idle(self):
        r = PPStitcher(
            stage_fwd_us={s: 100 for s in range(4)},
            stage_bwd_us={s: 200 for s in range(4)},
            pp=4, M=8, p2p_latency_us=0, schedule="1f1b",
        ).stitch()
        bn, _, _ = _bottleneck_idle(r)
        assert r.bubble_us == pytest.approx(bn, abs=1)

    def test_zb_bubble_equals_bottleneck_idle(self):
        r = PPStitcher(
            stage_fwd_us={s: 100 for s in range(4)},
            stage_bwd_us={s: 200 for s in range(4)},
            stage_bwd_dw_us={s: 80 for s in range(4)},
            pp=4, M=8, p2p_latency_us=0, schedule="zb",
        ).stitch()
        bn, _, _ = _bottleneck_idle(r)
        assert r.bubble_us == pytest.approx(bn, abs=1)

    def test_zb_zero_dw_bubble_equals_bottleneck_idle(self):
        r = PPStitcher(
            stage_fwd_us={s: 100 for s in range(4)},
            stage_bwd_us={s: 200 for s in range(4)},
            stage_bwd_dw_us={s: 0 for s in range(4)},
            pp=4, M=8, p2p_latency_us=0, schedule="zb",
        ).stitch()
        bn, _, _ = _bottleneck_idle(r)
        assert r.bubble_us == pytest.approx(bn, abs=1)

    def test_vpp_bubble_equals_bottleneck_idle(self):
        r = PPStitcher(
            stage_fwd_us={s: 400 for s in range(2)},
            stage_bwd_us={s: 800 for s in range(2)},
            pp=2, M=8, p2p_latency_us=0, schedule="interleaved",
            vpp_chunks=4,
        ).stitch()
        bn, _, _ = _bottleneck_idle(r)
        assert r.bubble_us == pytest.approx(bn, abs=1)

    def test_pp1_zero_bubble_zero_idle(self):
        r = PPStitcher(
            stage_fwd_us={0: 100}, stage_bwd_us={0: 200},
            pp=1, M=4, schedule="1f1b",
        ).stitch()
        assert r.bubble_us == 0.0


class TestZeroBubbleConstraints:
    """ZeroBubble 三约束验证：F→B_dx、bwd_dx→bwd_dw、bwd_dx 跨 stage P2P。
    bwd_dw 不应有跨 stage P2P（权重梯度本地计算）。"""

    def _result(self, **kw):
        defaults = dict(
            stage_fwd_us={s: 100 for s in range(4)},
            stage_bwd_us={s: 200 for s in range(4)},
            stage_bwd_dw_us={s: 80 for s in range(4)},
            pp=4, M=8, p2p_latency_us=3, schedule="zb",
        )
        defaults.update(kw)
        return PPStitcher(**defaults).stitch()

    def test_f_to_bwd_dx_activation(self):
        """F[m] 完成后 bwd_dx[m] 才能开始（Edge ① 对 ZB 生效）。"""
        r = self._result()
        for t in r.tasks:
            if t.phase != "bwd_dx":
                continue
            fwd = [x for x in r.tasks if x.phase == "fwd"
                   and x.stage_id == t.stage_id and x.mb_id == t.mb_id]
            assert fwd, f"Missing fwd for {t.task_id}"
            fwd_end = fwd[0].start_us + fwd[0].latency_us
            assert t.start_us >= fwd_end - 0.01, \
                f"{t.task_id} starts {t.start_us:.1f} before fwd ends {fwd_end:.1f}"

    def test_bwd_dx_to_bwd_dw_chain(self):
        """bwd_dx[m] 完成后 bwd_dw[m] 才能开始（梯度链约束）。"""
        r = self._result()
        for t in r.tasks:
            if t.phase != "bwd_dw":
                continue
            dx = [x for x in r.tasks if x.phase == "bwd_dx"
                  and x.stage_id == t.stage_id and x.mb_id == t.mb_id]
            assert dx, f"Missing bwd_dx for {t.task_id}"
            dx_end = dx[0].start_us + dx[0].latency_us
            assert t.start_us >= dx_end - 0.01, \
                f"{t.task_id} starts {t.start_us:.1f} before bwd_dx ends {dx_end:.1f}"

    def test_bwd_dx_cross_stage_p2p(self):
        """跨 stage: bwd_dx[s+1][m] ⇢ bwd_dx[s][m]（反向 P2P）。"""
        r = self._result()
        for m in range(r.M):
            for s in range(r.pp - 1):
                src = [x for x in r.tasks if x.phase == "bwd_dx"
                       and x.stage_id == s + 1 and x.mb_id == m]
                dst = [x for x in r.tasks if x.phase == "bwd_dx"
                       and x.stage_id == s and x.mb_id == m]
                if src and dst:
                    src_end = src[0].start_us + src[0].latency_us
                    assert dst[0].start_us >= src_end - 0.01, \
                        f"s{s+1}→s{s} bwd_dx m{m}: src ends {src_end:.1f}, dst starts {dst[0].start_us:.1f}"

    def test_bwd_dw_no_cross_stage_p2p(self):
        """bwd_dw 依赖列表中不含跨 stage bwd_dw 依赖。
        权重梯度 dW = x^T · dy，x 和 dy 都在本 stage，dW 不跨 stage 传递。
        虽然同设备 bwd_dw 因 device-serial 自然排在上游之后，
        但 dependencies 列表中不应有跨 stage bwd_dw 条目。"""
        r = self._result()
        for t in r.tasks:
            if t.phase != "bwd_dw":
                continue
            cross_stage_dw_deps = [d for d in t.dependencies
                                   if d.endswith("_bwd_dw") and not d.startswith(f"s{t.stage_id}_")]
            assert not cross_stage_dw_deps, \
                f"{t.task_id} has cross-stage bwd_dw deps: {cross_stage_dw_deps}"

    def test_bwd_dw_floats_independently(self):
        """bwd_dw 不受 F[m+1] 约束，可自由填空闲（ZB 核心特性）。"""
        r = self._result()
        for t in r.tasks:
            if t.phase == "bwd_dw":
                fwd_next = [x for x in r.tasks if x.phase == "fwd"
                            and x.stage_id == t.stage_id and x.mb_id == t.mb_id + 1]
                dep_task_ids = t.dependencies
                if fwd_next:
                    assert fwd_next[0].task_id not in dep_task_ids, \
                        f"{t.task_id} should NOT depend on {fwd_next[0].task_id}"


class TestZeroBubbleNonUniform(TestZeroBubbleConstraints):
    """非均匀 stage 下 ZB 约束仍然成立。"""

    def _result(self, **kw):
        defaults = dict(
            stage_fwd_us={0: 100, 1: 150, 2: 80, 3: 120},
            stage_bwd_us={0: 200, 1: 300, 2: 160, 3: 240},
            stage_bwd_dw_us={0: 80, 1: 100, 2: 60, 3: 90},
            pp=4, M=8, p2p_latency_us=3, schedule="zb",
        )
        defaults.update(kw)
        return PPStitcher(**defaults).stitch()


class TestDualPipeVStreamFix:
    """DualPipeV _vstage_to_device 修复：所有 stream_id ∈ {0..pp-1}。"""

    def test_all_streams_are_physical_devices(self):
        r = PPStitcher(
            stage_fwd_us={s: 400 for s in range(2)},
            stage_bwd_us={s: 800 for s in range(2)},
            pp=2, M=8, p2p_latency_us=5, schedule="dualpipev",
            vpp_chunks=4,
        ).stitch()
        stream_ids = set(t.stream_id for t in r.tasks)
        assert stream_ids.issubset({0, 1}), \
            f"stream_ids={stream_ids} should only contain 0,1 (pp=2)"

    def test_dualpipev_bubble_equals_bottleneck_idle(self):
        r = PPStitcher(
            stage_fwd_us={s: 400 for s in range(2)},
            stage_bwd_us={s: 800 for s in range(2)},
            pp=2, M=8, p2p_latency_us=0, schedule="dualpipev",
            vpp_chunks=4,
        ).stitch()
        bn, _, _ = _bottleneck_idle(r)
        assert r.bubble_us == pytest.approx(bn, abs=5)

    def test_dualpipev_pp3_streams(self):
        r = PPStitcher(
            stage_fwd_us={s: 300 for s in range(3)},
            stage_bwd_us={s: 600 for s in range(3)},
            pp=3, M=8, p2p_latency_us=5, schedule="dualpipev",
            vpp_chunks=4,
        ).stitch()
        stream_ids = set(t.stream_id for t in r.tasks)
        assert stream_ids.issubset({0, 1, 2})


class TestStepTimeDecomposition:
    """warmup + steady + cooldown = step_time 对所有策略成立。"""

    def test_1f1b(self):
        r = PPStitcher(
            stage_fwd_us={s: 100 for s in range(4)},
            stage_bwd_us={s: 200 for s in range(4)},
            pp=4, M=8, p2p_latency_us=5, schedule="1f1b",
        ).stitch()
        assert pytest.approx(r.step_time_us, abs=1) == r.warmup_us + r.steady_us + r.cooldown_us

    def test_zb(self):
        r = PPStitcher(
            stage_fwd_us={s: 100 for s in range(4)},
            stage_bwd_us={s: 200 for s in range(4)},
            stage_bwd_dw_us={s: 80 for s in range(4)},
            pp=4, M=8, p2p_latency_us=5, schedule="zb",
        ).stitch()
        assert pytest.approx(r.step_time_us, abs=1) == r.warmup_us + r.steady_us + r.cooldown_us

    def test_vpp(self):
        r = PPStitcher(
            stage_fwd_us={s: 400 for s in range(2)},
            stage_bwd_us={s: 800 for s in range(2)},
            pp=2, M=8, p2p_latency_us=5, schedule="interleaved",
            vpp_chunks=4,
        ).stitch()
        assert pytest.approx(r.step_time_us, abs=1) == r.warmup_us + r.steady_us + r.cooldown_us


class TestNonUniformStages:
    """非均匀 stage：公式 bubble = step - M*max(fwd+bwd)。

    非均匀时瓶颈 stage 空闲可能小于其他 stage 空闲，
    因为慢 stage 拖长了 step_time 使快 stage 更空。
    公式 bubble 仍正确度量"超出瓶颈理想的部分"。
    """

    def test_1f1b_bubble_nonnegative(self):
        """1F1B 非均匀: bubble >= 0。per_stage = max(150+300)=450。
        ideal = 8*450=3600, step > 3600（有气泡）。
        """
        r = PPStitcher(
            stage_fwd_us={0: 100, 1: 150, 2: 80, 3: 120},
            stage_bwd_us={0: 200, 1: 300, 2: 160, 3: 240},
            pp=4, M=8, p2p_latency_us=5, schedule="1f1b",
        ).stitch()
        assert r.bubble_us >= 0.0
        assert r.bubble_us == pytest.approx(r.step_time_us - 8 * 450, abs=5)

    def test_zb_bubble_nonnegative(self):
        """ZB 非均匀: per_stage = max(fwd+bwd) = 450 (stage 1)。
        bubble = step - 8*450 >= 0。
        """
        r = PPStitcher(
            stage_fwd_us={0: 100, 1: 150, 2: 80, 3: 120},
            stage_bwd_us={0: 200, 1: 300, 2: 160, 3: 240},
            stage_bwd_dw_us={0: 80, 1: 100, 2: 60, 3: 90},
            pp=4, M=8, p2p_latency_us=5, schedule="zb",
        ).stitch()
        assert r.bubble_us >= 0.0
        per_stage = max(100 + 200, 150 + 300, 80 + 160, 120 + 240)
        assert r.bubble_us == pytest.approx(r.step_time_us - 8 * per_stage, abs=5)