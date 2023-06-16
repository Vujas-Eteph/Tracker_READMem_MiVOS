# Decorators for measuring the executions time

# by Stéphane Vujasinovic

from functools import wraps
import time


def khronos_metron(func):   # Credit: https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
    @wraps(func)
    def khronos_metron_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # Return to a seperate log, so that this can be processed at an ulterior stage, by a diff. prog.
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return khronos_metron_wrapper
