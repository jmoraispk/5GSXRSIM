"""Packet."""

from typing import Any

from dpkt.ethernet import Ethernet
from dpkt.ip import IP
from dpkt.udp import UDP
from dpkt.rtp import RTP


class Packet:
    def __init__(self, timestamp: int, buffer: Any) -> None:
        self._timestamp = timestamp
        self._buffer = buffer

        # Assume only IP>UDP>RTP packets in the trace
        self._eth = Ethernet(self._buffer)
        self._ip: IP = self._eth.data
        self._udp: UDP = self._ip.data

        self._rtp = RTP()
        self._rtp.unpack(self._udp.data)

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def size(self) -> int:
        return len(self._ip)

    @property
    def buffer(self) -> Any:
        return self._buffer

    @property
    def rtp_timestamp(self) -> int:
        return self._rtp.ts
