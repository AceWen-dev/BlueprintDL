from dataclasses import FrozenInstanceError

import pytest

from dlkit.registry import ComponentRecord, Registry, iter_component_records


class _DirectComponent:
    pass


def test_direct_registration_records_component_provenance():
    registry = Registry('provenance-direct')
    source_tags = ['vision', 'builtin']

    returned = registry.register(
        'direct',
        _DirectComponent,
        provider='dlkit',
        tags=source_tags,
        description='A component registered directly.',
    )
    source_tags.append('mutated-after-registration')

    record = registry.describe('direct')
    assert returned is _DirectComponent
    assert isinstance(record, ComponentRecord)
    assert record.registry == 'provenance-direct'
    assert record.name == 'direct'
    assert record.target is _DirectComponent
    assert record.provider == 'dlkit'
    assert record.module == __name__
    assert record.qualname == '_DirectComponent'
    assert record.tags == ('vision', 'builtin')
    assert record.description == 'A component registered directly.'

    with pytest.raises(FrozenInstanceError):
        record.provider = 'other'

    registry.unregister('direct')


def test_decorator_registration_accepts_metadata_and_defaults():
    registry = Registry('provenance-decorator')

    @registry.register(
        provider='third-party',
        tags='experimental',
        description='Registered with decorator syntax.',
    )
    class DecoratedComponent:
        pass

    record = registry.describe('DecoratedComponent')
    assert registry.get('DecoratedComponent') is DecoratedComponent
    assert record.target is DecoratedComponent
    assert record.provider == 'third-party'
    assert record.tags == ('experimental',)
    assert record.description == 'Registered with decorator syntax.'

    registry.register('without-metadata', _DirectComponent)
    default_record = registry.describe('without-metadata')
    assert default_record.provider is None
    assert default_record.tags == ()
    assert default_record.description is None

    registry.unregister('DecoratedComponent')
    registry.unregister('without-metadata')


def test_records_and_global_iterator_filter_by_provider_and_registry():
    first = Registry('provenance-first')
    second = Registry('provenance-second')

    first.register('owned-a', _DirectComponent, provider='provider-a')
    first.register('owned-b', _DirectComponent, provider='provider-b')
    second.register('also-owned-a', _DirectComponent, provider='provider-a')

    assert [record.name for record in first.records()] == ['owned-a', 'owned-b']
    assert [record.name for record in first.records(provider='provider-a')] == [
        'owned-a'
    ]

    by_object = list(
        iter_component_records(provider='provider-a', registry=first)
    )
    by_name = list(
        iter_component_records(
            provider='provider-a', registry='provenance-second'
        )
    )
    all_for_provider = list(iter_component_records(provider='provider-a'))

    assert [record.name for record in by_object] == ['owned-a']
    assert [record.name for record in by_name] == ['also-owned-a']
    assert {record.name for record in all_for_provider} >= {
        'owned-a',
        'also-owned-a',
    }

    first.unregister('owned-a')
    first.unregister('owned-b')
    second.unregister('also-owned-a')


def test_unregister_removes_target_and_provenance_record():
    registry = Registry('provenance-unregister')
    registry.register('temporary', _DirectComponent, provider='temporary-provider')

    registry.unregister('temporary')

    assert 'temporary' not in registry
    assert registry.records() == []
    assert list(
        iter_component_records(
            provider='temporary-provider', registry=registry
        )
    ) == []
    with pytest.raises(KeyError):
        registry.get('temporary')
    with pytest.raises(KeyError):
        registry.describe('temporary')


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"provider": ""}, "provider"),
        ({"provider": 123}, "provider"),
        ({"tags": ["valid", 123]}, "tags"),
        ({"tags": [""]}, "tags"),
        ({"description": 123}, "description"),
    ],
)
def test_registration_rejects_invalid_provenance(metadata, message):
    registry = Registry("provenance-validation")

    with pytest.raises(TypeError, match=message):
        registry.register("invalid", _DirectComponent, **metadata)

    assert "invalid" not in registry


def test_dlkit_components_infer_core_provider_without_affecting_plugins():
    registry = Registry("provenance-core-inference")
    core_component = type(
        "CoreComponent",
        (),
        {"__module__": "dlkit.example_component"},
    )

    registry.register("core", core_component)
    registry.register("override", core_component, provider="project-a")

    assert registry.describe("core").provider == "blueprintdl"
    assert registry.describe("override").provider == "project-a"

    registry.unregister("core")
    registry.unregister("override")
