"""Program entrypoint."""

from argparse import ArgumentParser, Namespace

from pcap_traces.main import main


def _parse_args() -> Namespace:
    parser = ArgumentParser(prog="python3 -m pcap_traces",
                            description="Reads and process PCAP traces containing RTP streams")

    parser.add_argument('--pcap', action='store', type=str, required=True, help="Input PCAP file")
    parser.add_argument('--output', action='store', type=str, required=True, help="Output PCAP file")

    args = parser.parse_args()
    return args


cli_args = _parse_args()
main(cli_args)
