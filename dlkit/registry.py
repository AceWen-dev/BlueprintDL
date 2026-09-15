from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentRecord:
    """Immutable provenance metadata for one registered component."""

    registry: str
    name: str
    target: object
    provider: str | None
    module: str
    qualname: str
    tags: tuple[str, ...]
    description: str | None


class Registry:  # 这个类Registry
    def __init__(self, name):  #self其实就是为了类的实例化诞生的中间态占位
        self._name = name  #赋予属性
        self._items = {}   #这个下划线只是给读者提醒不要碰,这个注册表的核心就是往这个字典里面放入各种类
        self._records = {}
        _ALL_REGISTRIES.append(self) #把类的实例塞入列表

    @property #把方法变成属性dog.name()->dog.name
    def name(self):  
        return self._name

    def register(
        self,
        name=None,
        item=None,
        *,
        provider=None,
        tags=None,
        description=None,
    ): #闭包，用于装饰器，装饰器注册的核心机制
        if item is not None:
            self._register(
                name,
                item,
                provider=provider,
                tags=tags,
                description=description,
            )
            return item

        def _decorator(obj):
            self._register(
                name,
                obj,
                provider=provider,
                tags=tags,
                description=description,
            )
            return obj

        return _decorator

    def _register(self, name, item, *, provider=None, tags=None, description=None):
        # 确定注册所用的键名：优先用传入的 name，否则自动取类的 __name__
        key = name or getattr(item, '__name__', None)#getattr是获得item得属性，item就是实现模块功能的函数也是对象有默认属性
        if not key:
            raise ValueError(
                'Could not infer a name for %r; pass name= explicitly' % (item,)
            )
        # 防止重复注册同一个名字
        if key in self._items:
            raise KeyError(
                '%r is already registered in registry %r' % (key, self._name)
            )
        if provider is not None and (
            not isinstance(provider, str) or not provider.strip()
        ):
            raise TypeError('provider must be a non-empty string or None')
        if description is not None and not isinstance(description, str):
            raise TypeError('description must be a string or None')
        if tags is None:
            normalized_tags = ()
        elif isinstance(tags, str):
            normalized_tags = (tags,)
        else:
            normalized_tags = tuple(tags)
        if any(not isinstance(tag, str) or not tag for tag in normalized_tags):
            raise TypeError('tags must contain only non-empty strings')

        module = getattr(item, '__module__', None) or type(item).__module__
        if not isinstance(module, str):
            module = type(item).__module__
        qualname = (
            getattr(item, '__qualname__', None)
            or getattr(item, '__name__', None)
            or type(item).__qualname__
        )
        if not isinstance(qualname, str):
            qualname = type(item).__qualname__
        resolved_provider = provider.strip() if provider is not None else None
        if resolved_provider is None and (
            module == 'dlkit' or module.startswith('dlkit.')
        ):
            resolved_provider = 'blueprintdl'
        record = ComponentRecord(
            registry=self._name,
            name=key,
            target=item,
            provider=resolved_provider,
            module=module,
            qualname=qualname,
            tags=normalized_tags,
            description=description,
        )
        # 存入字典：键=名字，值=类/函数
        self._items[key] = item
        self._records[key] = record

    def unregister(self, name):
        self._items.pop(name, None)
        self._records.pop(name, None)

    def get(self, name):
        if name not in self._items:
            raise KeyError(
                '%r not found in registry %r. Available: %s'
                % (name, self._name, sorted(self._items))
            )
        return self._items[name]

    def keys(self):
        return list(self._items)

    def values(self):
        return list(self._items.values())

    def describe(self, name):
        if name not in self._records:
            self.get(name)
        return self._records[name]

    def records(self, provider=None):
        records = list(self._records.values())
        if provider is None:
            return records
        return [record for record in records if record.provider == provider]

    def __contains__(self, name):
        return name in self._items

    def __repr__(self):
        return 'Registry(%r, %d items)' % (self._name, len(self._items))


_ALL_REGISTRIES = []
#以下都是Registry类的实例化，它们的属性item是字典里面装了各种类/函数
BACKBONES = Registry('backbones')
DECODERS = Registry('decoders')
HEADS = Registry('heads')
MODELS = Registry('models')
LOSSES = Registry('losses')
DATASETS = Registry('datasets')
TRANSFORMS = Registry('transforms')
CLEANERS = Registry('cleaners')
METRICS = Registry('metrics')
OPTIMIZERS = Registry('optimizers')
SCHEDULERS = Registry('schedulers')


def iter_component_records(provider=None, registry=None):
    """Iterate component records, optionally filtered by provider and registry."""

    for current in _ALL_REGISTRIES:
        if isinstance(registry, Registry):
            if current is not registry:
                continue
        elif registry is not None and current.name != registry:
            continue
        yield from current.records(provider=provider)


def _resolve(type_name):#负责把配置字典的字符串名字，对应到注册表中的具体类（说白话就是把配置的字符串变成真正的类）
    if isinstance(type_name, str) and '.' in type_name:
        reg_name, cls_name = type_name.split('.', 1)
        for registry in _ALL_REGISTRIES:
            if registry.name == reg_name:
                return registry.get(cls_name)
        raise KeyError('Unknown registry name %r in qualified type %r' % (reg_name, type_name))

    matches = [r for r in _ALL_REGISTRIES if type_name in r]
    if not matches:
        raise KeyError(

            '%r is not registered in any registry. Available: %s'
            % (type_name, sorted({k for r in _ALL_REGISTRIES for k in r.keys()}))
        )
    if len(matches) > 1:
        names = ', '.join('%s.%s' % (r.name, type_name) for r in matches)
        raise KeyError(
            '%r is ambiguous, found in %d registries. Use a qualified name: %s'
            % (type_name, len(matches), names)
        )
    return matches[0].get(type_name)


def build_from_cfg(cfg): #函数的输入是字典或json格式的配置文件，输出是从这个写配置文件中提取的类的实例化对象
    if isinstance(cfg, (list, tuple)):
        return [build_from_cfg(c) for c in cfg]
    if not isinstance(cfg, dict):
        return cfg
    if 'type' in cfg:
        item = _resolve(cfg['type'])
        params = cfg.get('params', {})
        if getattr(item, '_manual_build', False):
            return item(**params)
        params = build_from_cfg(params)
        return item(**params) #返回带参数的类
    return {k: build_from_cfg(v) for k, v in cfg.items()}#item返回键值对，{}里面是字典推导式，[]是列表推导式
