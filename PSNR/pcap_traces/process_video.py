
from argparse import ArgumentParser, Namespace
import sys
sys.path.append('../')
from pcap_traces.main import main
import ffmpeg
import subprocess
import os
from ffmpeg_quality_metrics import FfmpegQualityMetrics
import pandas as pd

def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Gets a video file and a PCAP from streamed video and dumps PSNR between them")

    parser.add_argument('--pcap', action='store', type=str, required=True, help="Input PCAP file")
    parser.add_argument('--output', action='store', type=str, required=False, help="Output PCAP file, optional") # this is only kept so pcap-traces module doesnt need modification
    parser.add_argument('--video', action='store', type=str, required=True, help="Input video file")
    parser.add_argument('--verbose', action='store_true', help="stdout print per packet")
    args = parser.parse_args()
    return args


cli_args = _parse_args()
cli_args.output = "temp.pcap"
main(cli_args)

# gst part
temp_video_name = "tempvideo.yuv"
gst_cmd = 'gst-launch-1.0 filesrc location="{trace}" ! pcapparse caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! filesink location="{output}"'.format(trace=cli_args.output, output=temp_video_name)
result = subprocess.run(gst_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
os.remove(cli_args.output)

# ffmpeg convert
converted_file_name = "converted.mp4"
convert_cmd = 'ffmpeg -s 1920x1080 -i {input} -ss 00:00:00 -c:v libx264 -s:v 1920x1080 -t 00:01:00 {converted}'.format(input=temp_video_name,converted=converted_file_name)
result = subprocess.run(convert_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
os.remove(temp_video_name)

# ffmpeg get PSNR with shell
#psnr_file_name='psnr.txt'
#psnr_cmd = 'ffmpeg -i {modified} -i {original} -lavfi psnr=stats_file={psnr_logfile} -f null -'.format(modified=converted_file_name, original=cli_args.video,psnr_logfile=psnr_file_name)
#result = subprocess.run(psnr_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
#os.remove(converted_file_name)

# get PSNR with ffmpeg-quality-metrics

ffqm = FfmpegQualityMetrics(converted_file_name, cli_args.video)
temp = ffqm.calc(["ssim", "psnr"])
df = pd.DataFrame.from_dict(temp['psnr'])
os.remove(converted_file_name)

df.to_csv("psnr.csv",index=False)



