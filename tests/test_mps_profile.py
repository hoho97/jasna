from contextlib import contextmanager
import threading

import pytest

from jasna.mps_profiling import WallProfile, ObservedRLock


def test_nested_exclusive_wall_and_exception():
    times = iter([0., 1., 3., 5.])
    profile = WallProfile(lambda: next(times))
    with pytest.raises(ValueError), profile.measure('outer', units=10):
        with profile.measure('child'):
            raise ValueError('test')
    rows = profile.snapshot()
    assert rows['outer'] == dict(calls=1, inclusive_seconds=5., exclusive_seconds=3., units=10, calls_per_second=.2, units_per_second=2.)
    assert rows['child']['exclusive_seconds'] == 2.


def test_reentrant_lock_records_only_outer_hold_and_releases_on_error():
    profile = WallProfile()
    underlying = threading.RLock()
    observed = ObservedRLock(underlying, profile)
    with pytest.raises(ValueError), observed:
        with observed:
            raise ValueError('test')
    rows = profile.snapshot()
    assert rows['lock.hold/MainThread']['calls'] == 1
    assert rows['lock.wait/MainThread']['calls'] == 1
    acquired = []
    def worker():
        with observed:
            acquired.append(True)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive() and acquired == [True]


def test_threads_have_independent_interval_stacks():
    profile = WallProfile()
    barrier = threading.Barrier(2)
    def worker():
        with profile.measure('work', units=4):
            barrier.wait(timeout=2)
    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    row = profile.snapshot()['work']
    assert row['calls'] == 2 and row['units'] == 8
    assert row['inclusive_seconds'] == row['exclusive_seconds']


def test_observer_restores_methods_and_lock_on_failure():
    from jasna import accelerator
    from jasna.mps_profiling import observe_pipeline
    from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
    original = RfDetrMosaicDetectionModel._postprocess
    lock = accelerator._MPS_EXECUTION_LOCK
    with pytest.raises(ValueError), observe_pipeline(WallProfile()):
        assert isinstance(vars(RfDetrMosaicDetectionModel)['_postprocess'], staticmethod)
        assert accelerator._MPS_EXECUTION_LOCK is not lock
        raise ValueError('test')
    assert RfDetrMosaicDetectionModel._postprocess is original
    assert accelerator._MPS_EXECUTION_LOCK is lock


def test_transfer_detail_attributes_bytes_without_double_counting():
    import torch
    from jasna.mps_profiling import observe_pipeline
    tensor = torch.ones(2, 3)
    profile = WallProfile()
    with observe_pipeline(profile, detail_transfers=True):
        assert tensor.to('meta').shape == (2, 3)
    rows = profile.snapshot()
    aggregate = rows['transfer/cpu->meta']
    detail = rows['transfer_detail/MainThread/cpu->meta/torch.float32/(2, 3)']
    assert aggregate['calls'] == detail['calls'] == 1
    assert aggregate['units'] == detail['units'] == 24
    assert aggregate['inclusive_seconds'] >= detail['inclusive_seconds']
    assert aggregate['exclusive_seconds'] < aggregate['inclusive_seconds']
