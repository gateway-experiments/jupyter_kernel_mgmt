import os
import pytest

from ipykernel.kernelspec import make_ipkernel_cmd
from jupyter_kernel_mgmt.subproc.launcher import (
    SubprocessKernelLauncher, start_new_kernel
)

TIMEOUT = 10

pytestmark = pytest.mark.asyncio


async def test_get_connect_info(asyncio_patch):
    launcher = SubprocessKernelLauncher(make_ipkernel_cmd(), os.getcwd())
    info, km = await launcher.launch()
    try:
        assert set(info.keys()) == {
            'ip', 'transport',
            'hb_port', 'shell_port', 'stdin_port', 'iopub_port', 'control_port',
            'key', 'signature_scheme',
        }
    finally:
        await km.kill()
        await km.cleanup()


async def test_start_new_kernel(asyncio_patch):
    km, kc = await start_new_kernel(make_ipkernel_cmd(), startup_timeout=TIMEOUT)
    try:
        assert await km.is_alive()
        assert await kc.is_alive()
    finally:
        kc.shutdown_or_terminate()
        kc.close()
        await km.kill()
