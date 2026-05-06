"""Telemetry stubs for the SP-Mem artifact.

The artifact does not send telemetry. These no-op functions preserve the internal
interfaces used by the memory implementation.
"""

SPMEM_TELEMETRY = False


def capture_event(*args, **kwargs):
    return None


def capture_client_event(*args, **kwargs):
    return None
