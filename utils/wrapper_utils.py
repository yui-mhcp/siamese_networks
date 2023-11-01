# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import functools

from utils.generic_utils import get_args, signature_to_str

def dispatch_wrapper(methods, name, default = None):
    def wrapper(fn):
        def dispatch(dispatch_fn = None, keys = None):
            if dispatch_fn is not None and callable(dispatch_fn):
                if keys is None: keys = dispatch_fn.__name__.split('_')[-1]
                if not isinstance(keys, (list, tuple)): keys = [keys]
                methods.update({k : dispatch_fn for k in keys})
                
                add_dispatch_doc(
                    fn = fn, dispatch_fn = dispatch_fn, keys = keys, name = name, default = default
                )
                return dispatch_fn
            
            keys = dispatch_fn
            return lambda dispatch_fn: fn.dispatch(dispatch_fn, keys)
        
        fn.dispatch = dispatch
        fn.methods  = methods
        
        for method_name, method_fn in methods.items():
            fn.dispatch(method_fn, method_name)
        
        return fn
    
    return wrapper

def add_dispatch_doc(fn, dispatch_fn, name, keys, show_doc = False, default = None):
    if not keys: return
    display = keys[0] if len(keys) == 1 else tuple(keys)
    
    fn.__doc__ = '{}{}{}: {}\n    {}{}{}'.format(
        '{}\n\n'.format(fn.__doc__) if fn.__doc__ is not None else '',
        name,
        ' (default) ' if default and default in keys else ' ',
        display,
        dispatch_fn.__name__,
        inspect.signature(dispatch_fn),
        '\n{}'.format(dispatch_fn.__doc__) if show_doc and dispatch_fn.__doc__ else ''
    )

def partial(fn = None, * partial_args, _force = False, ** partial_config):
    """
        Wraps `fn` with default args or kwargs. It acts similarly to `functools.wraps` but gives cleaner doc information
        The major difference with `functools.partial` is that it supports class methods !
        
        Arguments :
            - fn    : the function to call
            - * partial_args    : values for the first positional arguments
            - _force    : whether to force kwargs `partial_config` or not
            - ** partial_config : new default values for keyword arguments
        Return :
            - wrapped function (if `fn`) or a decorator
        
        Note that partial positional arguments are only usable in the non-decorator mode (which is also the case in `functools.partial`)
        
        /!\ IMPORTANT : the below examples can be executed with `functools.partial` and will give the exact same behavior
        
        Example :
        ```python
        @partial(b = 3)
        def foo(a, b = 2):
            return a ** b
        
        print(foo(2))   # displays 8 as `b`'s default value is now 3
        
        # In this case, the `a` parameter is forced to be 2 and will be treated as a constant value
        # passing positional arguments to `partial` is a bit risky, but it can be useful in some scenario if you want to *remove* some parameters and fix them to a constant
        # Example : `logging.dev = partial(logging.log, DEV)` --> `logging.dev('message')`
        # Example : `logging.Logger.dev = partial(logging.Logger.log, DEV)`
        # --> `logging.getLogger(__name__).dev('message')` This will not work with functools.partial
        foo2 = partial(foo, 2)

        print(foo2())   # displays 8 as `a`'s value has been set to 2
        print(foo2(b = 4)) # displays 16
        print(foo2(4))  # raises an exception because `b` is given as positional arguments (4) and keyword argument (3)
        
        # In this case, foo3 **must** be called with kwargs, otherwise it will raise an exception
        foo3 = partial(a = 2, b = 2)
        ```
    """
    def wrapper(fn):
        @functools.wraps(fn)
        def inner(* args, ** kwargs):
            if not add_self_first:
                args    = partial_args + args
            else:
                args    = args[:1] + partial_args + args[1:]
            kwargs  = {** kwargs, ** partial_config} if _force else {** partial_config, ** kwargs}
            return fn(* args, ** kwargs)
        
        inner.__doc__ = '{}{}{}'.format(
            '' if not partial_args else 'partial args : {} '.format(partial_args),
            '' if not partial_config else 'partial config : {}'.format(partial_config),
            '\n{}'.format(inner.__doc__) if inner.__doc__ else ''
        ).capitalize()
        
        inner.func  = fn
        inner.args  = partial_args
        inner.kwargs    = partial_config
        
        fn_arg_names    = get_args(fn)
        add_self_first  = fn_arg_names and fn_arg_names[0] == 'self'
        
        return inner
    
    return wrapper if fn is None else wrapper(fn)

def add_doc(original, wrapper = None, on_top = True):
    if wrapper is None: return lambda fn: add_doc(original, fn, on_top)
    if wrapper.__doc__ is None: return wrapper
    
    if not original.__doc__:
        original.__doc__ = wrapper.__doc__
    elif on_top:
        original.__doc__ = '{}\n\n{}'.format(wrapper.__doc__, original.__doc__)
    else:
        original.__doc__ = '{}\n\n{}'.format(original.__doc__, wrapper.__doc__)
    
    return wrapper

def insert_signature(* args, ** kwargs):
    """ Equivalent to `format_doc` but it allows positional arguments (must have a `__name__`) """
    kwargs.update({
        arg.__name__ : signature_to_str(arg) for arg in args
    })
    return format_doc(** kwargs)

def format_doc(fn = None, ** kwargs):
    """ Formats the `fn.__doc__` with the given `kwargs` (keys must be in `fn.__doc__` between {}) """
    def wrapper(fn):
        if fn.__doc__: fn.__doc__ = fn.__doc__.format(** {
            k : v if not callable(v) else signature_to_str(v) for k, v in kwargs.items()
        })
        return fn
    return wrapper if fn is None else wrapper(fn)

def fake_wrapper(* args, ** kwargs):
    if len(args) == 1 and callable(args[0]): return args[0]
    return lambda fn: fn
