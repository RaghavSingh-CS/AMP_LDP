"""Validation-only network isolation. Opt in by prepending this directory to PYTHONPATH.

Applied before Streamlit/preflight imports in fresh child processes. Localhost
connections remain available for health/browser tests. No system settings change.
"""

import socket

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_getaddrinfo = socket.getaddrinfo
_sendto = socket.socket.sendto


def require_loopback(address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise OSError(f"Offline demo validation blocked outbound network: {host}")


def connect(self, address):
    require_loopback(address)
    return _connect(self, address)


def connect_ex(self, address):
    require_loopback(address)
    return _connect_ex(self, address)


def getaddrinfo(host, *args, **kwargs):
    require_loopback(host)
    return _getaddrinfo(host, *args, **kwargs)


def sendto(self, data, *args):
    require_loopback(args[-1])
    return _sendto(self, data, *args)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.getaddrinfo = getaddrinfo
socket.socket.sendto = sendto
