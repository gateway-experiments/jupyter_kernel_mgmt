"""Tools for managing kernel specs"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import io
import json
import logging
import os
import re
import shutil
import warnings

from .discovery import KernelProviderBase
from .subproc import SubprocessKernelLauncher

from ipython_genutils.importstring import import_item
from jupyter_core.paths import jupyter_data_dir, jupyter_path, SYSTEM_JUPYTER_PATH
from traitlets import HasTraits, List, Unicode, Dict, CaselessStrEnum
from traitlets.log import get_logger

try:
    from json import JSONDecodeError
except ImportError:
    # JSONDecodeError is new in Python 3.5, so while we support 3.4:
    JSONDecodeError = ValueError

pjoin = os.path.join
log = get_logger()

DEFAULT_SPEC_PROVIDER = 'spec'


class KernelSpecCacheEntry:
    """A class representing the kernel spec.

       Entries are stored in a dictionary corresponding to each kernel spec provider.  In
       addition to the resource directory and kernel.json (dict), the kernel.json's last
       modification time is also kept.  This value is used to determine stale entries.

       Note: files other than kernel.json are not cached.  This can change should we
       decide to store kernel specs in a non-filesystem data store, for example.
    """
    def __init__(self, name, resource_dir, last_mod_time, kernel_dict):
        self.name = name
        self.last_mod_time = last_mod_time
        self.spec = KernelSpec(resource_dir=resource_dir, **kernel_dict)

    def is_stale(self, current_mod_time):
        return current_mod_time > self.last_mod_time


class KernelSpecCache:
    """A singleton class used to cache kernel specs across multiple providers.

       The cache is implemented as a dictionary of dictionary, where the top-level dictionary
       is keyed by the kernel spec provider id.  The value of that entry is a dictionary of
       KernelSpecCacheEntry instances indexed by the kernel spec name.

       This cache is used to reduce the number of file opens necessary to maintain the
       existence of multiple KernelSpecProvider classes since the distinquishing characteristic
       is a provider id embedded in the kernel.json (with no embedded provider id representing
       the standard kernel spec).

       While cache entries are continually added, they are only removed upon attempts to use
       a cache entry that is no longer represented on the filesystem or via explict removal
       via the KernelSpecProvider.remove_kernel_spec entry.

       # TODO, we may want a periodic call back to look for missing entries.
    """

    __inst = None
    _providers = dict()

    @staticmethod
    def get_instance():
        if KernelSpecCache.__inst is None:
            KernelSpecCache()
        return KernelSpecCache.__inst

    def __init__(self):
        if KernelSpecCache.__inst is not None:
            raise RuntimeError("KernelSpecCache is a singleton, use KernelSpecCache.getInstance().")
        KernelSpecCache.__inst = self

    def get_entries(self, provider_id):
        """ Returns all KernelSpecCacheEntries for a given provider or an empty dict. """
        provider_cache = self._providers.get(provider_id)
        if provider_cache is not None:
            return provider_cache
        return {}

    def get_entry(self, provider_id, name):
        """ Returns the KernelSpecCacheEntry for the given provider and kernel spec name or None."""
        provider_cache = self._providers.get(provider_id)
        if provider_cache is not None:
            return provider_cache.get(name)
        return None

    def put_entry(self, provider_id, kernel_spec_entry):
        """ Addes the given KernelSpecCacheEntry for the given provider. """
        provider_cache = self._providers.get(provider_id)
        if provider_cache is None:
            provider_cache = dict()
            self._providers[provider_id] = provider_cache
        provider_cache[kernel_spec_entry.name] = kernel_spec_entry
        log.debug("KernelSpecCache.put_entry: provider: '{provider}', name: '{name}'".
                  format(provider=provider_id, name=kernel_spec_entry.name))

    def remove_entry(self, provider_id, name):
        """ Removes the cache entry for the given provider and kernel spec name. """
        provider_cache = self._providers.get(provider_id)
        if provider_cache is not None:
            provider_cache.pop(name)
            log.debug("KernelSpecCache.remove_entry: provider: '{provider}', name: '{name}'".
                      format(provider=provider_id, name=name))
        else:
            log.debug("KernelSpecCache.remove_entry: Not found for provider: '{provider}', name: '{name}'".
                      format(provider=provider_id, name=name))


class KernelSpec(HasTraits):
    argv = List()
    display_name = Unicode()
    language = Unicode()
    env = Dict()
    resource_dir = Unicode()
    interrupt_mode = CaselessStrEnum(
        ['message', 'signal'], default_value='signal'
    )
    metadata = Dict()

    @classmethod
    def from_resource_dir(cls, resource_dir):
        """Create a KernelSpec object by reading kernel.json

        Pass the path to the *directory* containing kernel.json.
        """
        kernel_file = pjoin(resource_dir, 'kernel.json')
        with io.open(kernel_file, 'r', encoding='utf-8') as f:
            kernel_dict = json.load(f)

        return cls(resource_dir=resource_dir, **kernel_dict)

    def to_dict(self):
        d = dict(argv=self.argv,
                 env=self.env,
                 display_name=self.display_name,
                 language=self.language,
                 interrupt_mode=self.interrupt_mode,
                 metadata=self.metadata)

        return d

    def to_json(self):
        """Serialise this kernelspec to a JSON object.

        Returns a string.
        """
        return json.dumps(self.to_dict())


_kernel_name_pat = re.compile(r'^[a-z0-9._\-]+$', re.IGNORECASE)


def _is_valid_kernel_name(name):
    """Check that a kernel name is valid."""
    return _kernel_name_pat.match(name)


_kernel_name_description = "Kernel names can only contain ASCII letters and numbers and these separators:" \
 " - . _ (hyphen, period, and underscore)."


def _is_kernel_dir(path):
    """Is ``path`` a kernel directory?"""
    return os.path.isdir(path) and os.path.isfile(pjoin(path, 'kernel.json'))


def _list_kernels_in(parent_dir):
    """Return a mapping of kernel names to resource directories from parent_dir.

    If parent_dir is None or does not exist, returns an empty dict.
    """
    if parent_dir is None or not os.path.isdir(parent_dir):
        return {}
    kernels = {}
    for f in os.listdir(parent_dir):
        path = pjoin(parent_dir, f)
        if not _is_kernel_dir(path):
            continue
        key = f.lower()
        if not _is_valid_kernel_name(key):
            warnings.warn("Invalid kernelspec directory name (%s): %s" % (_kernel_name_description, path), stacklevel=3)
        kernels[key] = path
    return kernels


class NoSuchKernel(KeyError):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "No such kernel named {}".format(self.name)


class KernelSpecProvider(KernelProviderBase):
    """Offers kernel types from installed kernelspec directories. """

    id = DEFAULT_SPEC_PROVIDER

    _kernel_spec_cache = KernelSpecCache.get_instance()

    def __init__(self, search_path=None):
        self.user_kernel_dir = self._user_kernel_dir_default()
        if search_path is None:
            search_path = self._kernel_dirs_default()
        self.kernel_dirs = search_path

    def find_kernels(self):
        for name, resource_dir in self.find_kernel_specs().items():
            cache_entry = self._kernel_spec_cache.get_entry(self.id, name)
            spec = cache_entry.spec

            log.debug("Found kernel %s in %s" % (name, resource_dir))

            yield name, {
                # TODO: get full language info
                'language_info': {'name': spec.language},
                'display_name': spec.display_name,
                'resource_dir': spec.resource_dir,
                'metadata': spec.metadata,
            }

    def launch(self, name, cwd=None):
        spec = self.get_kernel_spec(name)
        launcher = SubprocessKernelLauncher(kernel_cmd=spec.argv, extra_env=spec.env, cwd=cwd)
        return launcher.launch()

    def launch_async(self, name, cwd=None):
        from .subproc.async_manager import AsyncSubprocessKernelLauncher
        spec = self.get_kernel_spec(name)
        return AsyncSubprocessKernelLauncher(
            kernel_cmd=spec.argv, extra_env=spec.env, cwd=cwd).launch()

    @staticmethod
    def _user_kernel_dir_default():
        return pjoin(jupyter_data_dir(), 'kernels')

    @staticmethod
    def _kernel_dirs_default():
        dirs = jupyter_path('kernels')
        # TODO - why not now?
        # # At some point, we should stop adding .ipython/kernels to the path,
        # # but the cost to keeping it is very small.
        # try:
        #     from IPython.paths import get_ipython_dir
        # except ImportError:
        #     try:
        #         from IPython.utils.path import get_ipython_dir
        #     except ImportError:
        #         # no IPython, no ipython dir
        #         get_ipython_dir = None
        # if get_ipython_dir is not None:
        #     dirs.append(os.path.join(get_ipython_dir(), 'kernels'))
        return dirs

    def _validate_kernel_spec(self, name, resource_dir):
        """ Performs validation and (possible) caching of kernel.json file located in resource_dir.

            To minimize file opens, it first stats the file for last modification time and checks
            the cache if a) there's an entry for this provider and name and b) if the file has
            not been modified since cached.  If both are true, the current provider id is returned.

            If either check was false, the kernel.json file is loaded, checked for a provider_id
            entry, then cached corresponding to the found provider.  That provider id is returned,
            allowing the caller to discriminate entries for the current provider.
        """
        kernel_file = pjoin(resource_dir, 'kernel.json')
        try:
            last_mod_time = os.path.getmtime(kernel_file)

            # If we have a cache entry and its not stale - return, else, open file and
            # add to cache - irrespective of provider.
            cache_entry = self._kernel_spec_cache.get_entry(self.id, name)
            if cache_entry is not None and not cache_entry.is_stale(last_mod_time):
                return self.id

            with io.open(kernel_file, 'r', encoding='utf-8') as f:
                kernel_dict = json.load(f)
        except OSError as ose:
            log.warning("Failed to stat/open kernelspec in %s. Error was: %s", resource_dir, ose)
            return None
        except JSONDecodeError:
            log.warning("Failed to parse kernelspec in %s" % resource_dir)
            return None

        # Determine provider id.  If none is found, the default provider is assumed.
        provider_id = DEFAULT_SPEC_PROVIDER
        if 'metadata' in kernel_dict and 'provider_id' in kernel_dict['metadata']:
            provider_id = kernel_dict['metadata']['provider_id']

        # Check cache, if present, check mod time and update accordingly.
        cache_entry = self._kernel_spec_cache.get_entry(provider_id, name)
        if cache_entry is None or cache_entry.is_stale(last_mod_time):
            cache_entry = KernelSpecCacheEntry(name, resource_dir, last_mod_time, kernel_dict)
            self._kernel_spec_cache.put_entry(provider_id, cache_entry)

        return provider_id

    def find_kernel_specs(self):
        """Returns a dict mapping kernel names to resource directories for this kernel_spec provider."""
        d = {}
        for kernel_dir in self.kernel_dirs:
            kernels = _list_kernels_in(kernel_dir)
            for kname, resource_dir in kernels.items():
                provider_id = self._validate_kernel_spec(kname, resource_dir)
                if provider_id == self.id and kname not in d:
                    d[kname] = resource_dir
        return d

    def get_kernel_spec(self, kernel_name):
        """Returns a :class:`KernelSpec` instance for the given kernel_name.

        Raises :exc:`NoSuchKernel` if the given kernel name is not found.
        """
        d = self._kernel_spec_cache.get_entries(self.id)
        try:
            resource_dir = d[kernel_name.lower()].spec.resource_dir
        except KeyError:
            # May be that the kernelspec is newer than last cache update.  Use find_kernel_specs and try again.
            d = self.find_kernel_specs()
            try:
                resource_dir = d[kernel_name.lower()]
            except KeyError:
                raise NoSuchKernel(kernel_name)

        # Always return the kernel_spec from the file system.
        # Note: Notebook (and Lab) appear to fetch the set of kernelspecs just prior to launching
        # a kernel.  Since find_kernels is currently driven by the directory entries, missing (removed)
        # kernel specs will not be included, even though they remain in the cache.  As a result, in
        # those circumstances, a removed kernelspec will be detected prior to launching the kernel.
        # Nevertheless, let's trap OSError here and convert to NoSuchKernel.
        try:
            spec = KernelSpec.from_resource_dir(resource_dir)
        except OSError:
            # non-existent kernelspec dir, remove from cache
            self._kernel_spec_cache.remove_entry(self.id, kernel_name)
            raise NoSuchKernel(kernel_name)

        return spec

    def remove_kernel_spec(self, name):
        """Remove a kernel spec directory by name.

        Returns the path that was deleted.
        """
        kernel_spec_name = name.lower()
        specs = self.find_kernel_specs()
        spec_dir = specs[kernel_spec_name]
        log.debug("Removing %s", spec_dir)
        if os.path.islink(spec_dir):
            os.remove(spec_dir)
        else:
            shutil.rmtree(spec_dir)

        # Update cache
        self._kernel_spec_cache.remove_entry(self.id, kernel_spec_name)
        return spec_dir

    def _get_destination_dir(self, kernel_name, user=False, prefix=None):
        if user:
            return os.path.join(self.user_kernel_dir, kernel_name)
        elif prefix:
            return os.path.join(os.path.abspath(prefix), 'share', 'jupyter', 'kernels', kernel_name)
        else:
            return os.path.join(SYSTEM_JUPYTER_PATH[0], 'kernels', kernel_name)

    def install_kernel_spec(self, source_dir, kernel_name=None, user=False,
                            replace=None, prefix=None):
        """Install a kernel spec by copying its directory.

        If ``kernel_name`` is not given, the basename of ``source_dir`` will
        be used.

        If ``user`` is False, it will attempt to install into the systemwide
        kernel registry. If the process does not have appropriate permissions,
        an :exc:`OSError` will be raised.

        If ``prefix`` is given, the kernelspec will be installed to
        PREFIX/share/jupyter/kernels/KERNEL_NAME. This can be sys.prefix
        for installation inside virtual or conda envs.
        """
        source_dir = source_dir.rstrip('/\\')
        if not kernel_name:
            kernel_name = os.path.basename(source_dir)
        kernel_name = kernel_name.lower()
        if not _is_valid_kernel_name(kernel_name):
            raise ValueError("Invalid kernel name %r.  %s" % (kernel_name, _kernel_name_description))

        if user and prefix:
            raise ValueError("Can't specify both user and prefix. Please choose one or the other.")

        if replace is not None:
            warnings.warn(
                "replace is ignored. Installing a kernelspec always replaces an existing installation",
                DeprecationWarning,
                stacklevel=2,
            )

        destination = self._get_destination_dir(kernel_name, user=user, prefix=prefix)
        log.debug('Installing kernelspec in %s', destination)

        kernel_dir = os.path.dirname(destination)
        if kernel_dir not in self.kernel_dirs:
            log.warning("Installing to %s, which is not in %s. The kernelspec may not be found.",
                        kernel_dir, self.kernel_dirs)

        if os.path.isdir(destination):
            log.info('Removing existing kernelspec in %s', destination)
            shutil.rmtree(destination)

        shutil.copytree(source_dir, destination)
        log.info('Installed kernelspec %s in %s', kernel_name, destination)
        return destination


def _get_provider_instance(provider_class_name):
    if provider_class_name is None:
        return KernelSpecProvider()

    return import_item(provider_class_name)()


def find_kernel_specs(provider_class_name=None):
    """Returns a dict mapping kernel names to resource directories."""
    return _get_provider_instance(provider_class_name).find_kernel_specs()


def get_kernel_spec(kernel_name, provider_class_name=None):
    """Returns a :class:`KernelSpec` instance for the given kernel_name.

    Raises KeyError if the given kernel name is not found.
    """
    return _get_provider_instance(provider_class_name).get_kernel_spec(kernel_name)


def install_kernel_spec(source_dir, kernel_name=None, user=False, replace=False,
                        prefix=None, provider_class_name=None):
    return _get_provider_instance(provider_class_name).install_kernel_spec(source_dir,
                                                                           kernel_name, user, replace, prefix)


install_kernel_spec.__doc__ = KernelSpecProvider.install_kernel_spec.__doc__
