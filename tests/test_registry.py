import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlkit.registry import Registry, build_from_cfg


class _Dummy:
    pass


def test_register_and_get():
    reg = Registry('test')
    reg.register('foo', _Dummy)
    assert reg.get('foo') is _Dummy
    assert 'foo' in reg
    reg.unregister('foo')
    assert 'foo' not in reg


def test_decorator_register():
    reg = Registry('test')

    @reg.register()
    class Bar:
        pass

    assert reg.get('Bar') is Bar


def test_duplicate_raises():
    reg = Registry('test')
    reg.register('dup', _Dummy)
    try:
        reg.register('dup', _Dummy)
    except KeyError:
        return
    raise AssertionError('expected KeyError on duplicate registration')


def test_build_from_cfg_nested():
    reg = Registry('test')

    @reg.register()
    class Inner:
        def __init__(self, x=1):
            self.x = x

    @reg.register()
    class Outer:
        def __init__(self, inner, y=2):
            self.inner = inner
            self.y = y

    cfg = {
        'type': 'Outer',
        'params': {
            'inner': {'type': 'Inner', 'params': {'x': 10}},
            'y': 3,
        },
    }
    obj = build_from_cfg(cfg)
    assert isinstance(obj, Outer)
    assert isinstance(obj.inner, Inner)
    assert obj.inner.x == 10
    assert obj.y == 3


def test_qualified_name_resolution():
    reg = Registry('myreg')

    @reg.register()
    class Thing:
        pass

    obj = build_from_cfg({'type': 'myreg.Thing'})
    assert isinstance(obj, Thing)


def run_all():
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS %s' % name)
            except Exception as e:
                failed += 1
                print('FAIL %s: %s' % (name, e))
    print('done, %d failed' % failed)
    return failed


if __name__ == '__main__':
    sys.exit(run_all())
