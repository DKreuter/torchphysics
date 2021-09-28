import inspect
import numpy as np

class UserFunction:
    def __new__(cls, fun):
        if isinstance(fun, cls):
            return fun
        return super(UserFunction, cls).__new__(cls)

    def __init__(self, fun):
        self.fun = fun

        f_args = inspect.getfullargspec(self.fun).args

        # we check that the function defines all needed parameters
        if inspect.getfullargspec(self.fun).varargs is not None or \
            inspect.getfullargspec(self.fun).varkw is not None:
            raise ValueError("""
                             Variable arguments are not supported in
                             UserFunctions. Please use keyword arguments.
                             """)

        f_defaults = inspect.getfullargspec(self.fun).defaults
        f_kwonlyargs = inspect.getfullargspec(self.fun).kwonlyargs
        f_kwonlydefaults = inspect.getfullargspec(self.fun).kwonlydefaults
        # NOTE: By above check, there should not be kwonlyargs. However, we still catch
        # this case here.
        self.args = f_args + f_kwonlyargs

        # defaults always align at the end of the args
        self.defaults = {self.f_args[-i]: f_defaults[-i] for i in range(len(f_defaults),
                                                                        0,
                                                                        -1)}
        self.defaults.update(f_kwonlydefaults)

    def __call__(self,  vectorize=False, **args):
        # check that every necessary arg is given
        for key in self.necessary_args:
            assert key in args, \
                f"The argument '{key}' is necessary in {self.__name__} but not given."
        # if necessary, pass defaults
        inp = {key: args[key] for key in self.args if key in args}
        inp.update({key: self.defaults[key] for key in self.args if key not in args})
        if not vectorize:
            return self.fun(**inp)
        else:
            return self.apply_to_batch(inp)

    def apply_to_batch(self, inp):
        # apply the function to a batch of elements by running a for-loop

        # we assume that all inputs either have batch (i.e. maximum) dimension or
        # are a constant param
        batch_size = max(len(inp[key]) for key in inp)
        out = []
        for i in range(batch_size):
            inp_i = {}
            for key in inp:
                if len(inp[key]) == batch_size:
                    inp_i[key] = inp[key][i]
                else:
                    inp_i[key] = inp[key]
            o = self.fun(**inp_i)
            if o is not None:
                out.append(o)

    def __name__(self):
        return self.func.__name__

    def set_default(self, **args):
        self.defaults.update({key: args[key] for key in args if key in self.args})

    def remove_default(self, *args, **kwargs):
        for key in args:
            self.defaults.pop(key)
        for key in kwargs:
            self.defaults.pop(key)

    @property
    def necessary_args(self):
        return [arg for arg in self.args if arg not in self.defaults]

    @property
    def optional_args(self):
        return [arg for arg in self.args if arg in self.defaults]
