# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Borrowed from: https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/utils.py

import multiprocessing
import os
import signal
import sys
import time
import traceback
from typing import Optional

from .testing_util import run_test


def _failed_result(sample):
    return [-1 for _ in range(len(sample["inputs"]))]


def _temp_run(sample, generation, debug, conn, timeout):
    if hasattr(os, "setsid"):
        os.setsid()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    if devnull_fd > 2:
        os.close(devnull_fd)
    try:
        res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
        conn.send((res, metadata))
    except Exception:
        # Some tracebacks are extremely long. Keep child failures local to the sample.
        traceback.print_exc(10)
        conn.send((_failed_result(sample), {}))
    finally:
        conn.close()


def _terminate_process_tree(process, grace_period=1.0):
    if process.pid is None:
        return

    if process.is_alive():
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        except Exception:
            process.kill()

        deadline = time.monotonic() + grace_period
        while process.is_alive() and time.monotonic() < deadline:
            process.join(timeout=0.05)

    if process.is_alive():
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.join(timeout=1)
    else:
        process.join(timeout=1)


def check_correctness(in_outs: Optional[dict], generation, timeout=10, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`.
    """

    # Ray workers already run in a threaded runtime where libraries such as filelock
    # install fork-safety audit hooks. The default multiprocessing context is fork,
    # which can raise inside os.fork and make the reward actor unavailable. Spawn a
    # fresh interpreter and pass the result through a Pipe instead of Manager proxies.
    ctx_name = os.environ.get("PRIME_CODE_MP_CONTEXT", "spawn")
    ctx = multiprocessing.get_context(ctx_name)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_temp_run, args=(in_outs, generation, debug, child_conn, timeout))

    try:
        p.start()
    except Exception:
        child_conn.close()
        parent_conn.close()
        if debug:
            traceback.print_exc(10)
        return _failed_result(in_outs), []

    child_conn.close()
    try:
        if parent_conn.poll(timeout + 1):
            result, metadata = parent_conn.recv()
            return result, [metadata]

        if debug:
            print("global timeout")
        return _failed_result(in_outs), []
    finally:
        parent_conn.close()
        _terminate_process_tree(p, grace_period=1.0)
