
from argparse import ArgumentParser, Namespace
import sys
sys.path.append('../')
from main_pcap import main
# import ffmpeg
import subprocess
import os
from ffmpeg_quality_metrics import FfmpegQualityMetrics
import pandas as pd

def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Gets a video file and a PCAP from streamed video and dumps PSNR between them")
    
    # E.g: python process_video.py --params 'BW-150_E2E-LAT-50_LEN-16.0s_EDD_Offset-1.0_UE4' --seed 1 --e2e 50 
    #      --bitrate 100 --burst 0.6 --queue '10Q - 70.0%' 
       
    
    # E.g.: 'BW-150_E2E-LAT-50_LEN-16.0s_EDD_Offset-1.0_UE4'
    parser.add_argument('--params', action='store', type=str, required=True, help="Input Parameter String")
    
    # Up to 20 Seeds
    parser.add_argument('--seed', action='store', type=int, required=True, help="Number of Seeds")
    
    # E.g. '10Q - 70.0%'
    parser.add_argument('--queue', action='store', type=str, required=True, help="Parameters Queue Sim")
    
    # E2E Lat: 25/50/100 ms
    parser.add_argument('--e2e', action='store', type=int, required=True, help="E2E Latency")
    
    # Bitrate: E.g. 100  
    parser.add_argument('--bitrate', action='store', type=int, required=True, help="Trace Bitrate")
    
    #Burstiness: E.g. 0.6
    parser.add_argument('--burst', action='store', type=float, required=True, help="Trace Burstiness")
    
    # (Not needed?)    
    # parser.add_argument('--pcap', action='store', type=str, required=True, help="Input PCAP file")
    parser.add_argument('--output', action='store', type=str, required=False, help="Output PCAP file, optional") # this is only kept so pcap-traces module doesnt need modification
    
    # (Not needed? - SAME AS BITRATE)
    # parser.add_argument('--video', action='store', type=str, required=True, help="Input video file")
    
    parser.add_argument('--verbose', action='store_true', help="stdout print per packet")
    args = parser.parse_args()
    return args


cli_args = _parse_args()
cli_args.output = "temp.pcap"

og_video = os.getcwd() + f"\\PSNR\\PCAP_FILES\\input_APP{cli_args.bitrate}_16s.mp4" 


# PCAP PART: MAIN FROM PCAP-TRACES
n_ues = int(cli_args.params[-1])

seeds = cli_args.seed

for seed in range(1, seeds + 1):
    for ue in range(n_ues):
        main(cli_args, ue, n_ues, seed)

        # # Modify original pcap-trace
        
        # temp_pcap_name = "temp_output.pcap"
        # pcap_cmd = 'python -m pcap_traces --pcap input_0.pcap --output output_0.pcap > trace_0.csv'.format(
        #     trace=cli_args.output, output=temp_pcap_name)


        # gst part
        temp_video_name = "tempvideo.yuv"
        gst_cmd = 'gst-launch-1.0 filesrc location="{trace}" ! pcapparse caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! filesink location="{output}"'.format(
            trace=cli_args.output, output=temp_video_name)
        result = subprocess.run(gst_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        os.remove(cli_args.output)
        
        # ffmpeg convert
        
        # ADD: IF BITRATE = 50: FULL HD !!!
        #      ELSE: 4K RESOLUTION !!!
        # CHECK VIDEO TIMER!!!!!
        converted_file_name = "converted.mp4"
        if cli_args.bitrate == 50:
            convert_cmd = 'ffmpeg -s 1920x1080 -i {input} -ss 00:00:00 -c:v libx264 -s:v 1920x1080 -t 00:01:00 {converted}'.format(
            input=temp_video_name,converted=converted_file_name)
        else:     
            convert_cmd = 'ffmpeg -s 3840x2160 -i {input} -ss 00:00:00 -c:v libx264 -s:v 3840x2160 -t 00:01:00 {converted}'.format(
            input=temp_video_name,converted=converted_file_name)
        result = subprocess.run(convert_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        os.remove(temp_video_name)
        
        # ffmpeg get PSNR with shell
        #psnr_file_name='psnr.txt'
        #psnr_cmd = 'ffmpeg -i {modified} -i {original} -lavfi psnr=stats_file={psnr_logfile} -f null -'.format(modified=converted_file_name, original=cli_args.video,psnr_logfile=psnr_file_name)
        #result = subprocess.run(psnr_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        #os.remove(converted_file_name)
        
        # get PSNR with ffmpeg-quality-metrics
        
        ffqm = FfmpegQualityMetrics(converted_file_name, og_video)
        temp = ffqm.calc(["ssim", "psnr"])
        df = pd.DataFrame.from_dict(temp['psnr'])
        os.remove(converted_file_name)
        
        df.to_csv("psnr.csv",index=False)





