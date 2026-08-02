from __future__ import annotations

import argparse
import ipaddress
import logging
import select
import socket
import socketserver
import subprocess
import threading
from typing import Iterable


LOGGER = logging.getLogger("JPHConnectProxy")
MAX_HEADER_BYTES = 65_536


def select_ipv4(lines: Iterable[str]) -> str:
    """Return the first IPv4 address from ``dig +short`` output."""

    for line in lines:
        value = line.strip()
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            continue
        if address.version == 4:
            return value
    raise ValueError("DNS response contains no IPv4 address")


def parse_connect_target(target: bytes, allowed_ports: frozenset[int]) -> tuple[str, int]:
    """Parse a CONNECT authority and enforce the outbound port allow-list."""

    try:
        host_bytes, port_bytes = target.rsplit(b":", 1)
        host = host_bytes.decode("ascii")
        port = int(port_bytes)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("CONNECT target must be an ASCII host:port authority") from exc
    if not host or any(character.isspace() for character in host):
        raise ValueError("CONNECT host must be non-empty and contain no whitespace")
    if port not in allowed_ports:
        raise ValueError(f"CONNECT port {port} is not allowed")
    return host, port


class IPv4Resolver:
    def __init__(self, dns_server: str) -> None:
        ipaddress.ip_address(dns_server)
        self.dns_server = dns_server
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()

    def resolve(self, host: str) -> str:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None:
            if address.version != 4:
                raise ValueError("the proxy supports IPv4 destinations only")
            return str(address)

        with self._lock:
            cached = self._cache.get(host)
        if cached is not None:
            return cached
        result = subprocess.run(
            ["dig", "+short", "A", host, f"@{self.dns_server}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        resolved = select_ipv4(result.stdout.splitlines())
        with self._lock:
            self._cache[host] = resolved
        return resolved


class ConnectProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        resolver: IPv4Resolver,
        allowed_ports: frozenset[int],
    ) -> None:
        self.resolver = resolver
        self.allowed_ports = allowed_ports
        super().__init__(server_address, ConnectProxyHandler)


class ConnectProxyHandler(socketserver.BaseRequestHandler):
    server: ConnectProxyServer

    def _read_header(self) -> bytes:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = self.request.recv(4096)
            if not chunk:
                raise ConnectionError("client closed before sending a complete header")
            data += chunk
            if len(data) > MAX_HEADER_BYTES:
                raise ValueError("proxy request header is too large")
        return data

    def handle(self) -> None:
        self.request.settimeout(20)
        upstream: socket.socket | None = None
        established = False
        try:
            data = self._read_header()
            request_line = data.split(b"\r\n", 1)[0]
            method, target, _ = request_line.split(b" ", 2)
            if method != b"CONNECT":
                self.request.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                return
            host, port = parse_connect_target(target, self.server.allowed_ports)
            address = self.server.resolver.resolve(host)
            upstream = socket.create_connection((address, port), timeout=20)
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            established = True
            sockets = (self.request, upstream)
            while True:
                readable, _, _ = select.select(sockets, (), (), 60)
                for source in readable:
                    payload = source.recv(65_536)
                    if not payload:
                        return
                    destination = upstream if source is self.request else self.request
                    destination.sendall(payload)
        except Exception as exc:
            LOGGER.warning("proxy request failed: %s", exc)
            if not established:
                try:
                    self.request.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                except OSError:
                    pass
        finally:
            if upstream is not None:
                upstream.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Loopback-only HTTPS CONNECT proxy with an explicit DNS server"
    )
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18082)
    parser.add_argument("--dns-server", default="223.5.5.5")
    parser.add_argument("--allowed-port", type=int, action="append", default=[443])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.listen_host not in {"127.0.0.1", "::1"}:
        raise ValueError("the proxy may listen on loopback only")
    if not 1 <= args.listen_port <= 65_535:
        raise ValueError("listen port must be in [1, 65535]")
    allowed_ports = frozenset(args.allowed_port)
    if not allowed_ports or any(port < 1 or port > 65_535 for port in allowed_ports):
        raise ValueError("allowed ports must be in [1, 65535]")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    resolver = IPv4Resolver(args.dns_server)
    with ConnectProxyServer(
        (args.listen_host, args.listen_port), resolver, allowed_ports
    ) as server:
        LOGGER.info(
            "listening on %s:%d; DNS=%s; allowed_ports=%s",
            args.listen_host,
            args.listen_port,
            args.dns_server,
            sorted(allowed_ports),
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
