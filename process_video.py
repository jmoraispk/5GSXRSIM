
from argparse import ArgumentParser, Namespace
import sys
sys.path.append('../')
from main_pcap import main
# import ffmpeg
import subprocess
import os
import time
import pandas as pd
import numpy as np

from ffmpeg_quality_metrics import FfmpegQualityMetrics


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Gets a video file and a PCAP from streamed video and dumps PSNR between them")
    
    # E.g: 
    # python process_video.py --seed 1 --burst 0.6 --e2e 50 --queue '10Q - 70.0%' --bitrate 100 --params 'BW-150_E2E-LAT-50_LEN-16.0s_EDD_Offset-1.0_UE4' 
       
    
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
# E.g.: 'BW-150_E2E-LAT-50_LEN-16.0s_EDD_Offset-1.0_UE4'
sim_params = cli_args.params        
# Up to 20 Seeds
seed = cli_args.seed    
# E.g. '10Q - 70.0%'
queues = cli_args.queue # f"SEED1 - {args.queue} Load" # = args.queue    
# E2E Lat: 25/50/100 ms
e2e_lat = cli_args.e2e   
# Bitrate: E.g. 100  
bitrate = cli_args.bitrate    
#Burstiness: E.g. 0.6
burst = cli_args.burst


temp_file_name = f"APP{bitrate}-{burst}_E2E{e2e_lat}_{sim_params}_{queues}"

cli_args.output = f"temppcap_{temp_file_name}.pcap"

og_video = os.getcwd() + f"\\PSNR\\PCAP_FILES\\input_APP{cli_args.bitrate}.mp4" 

print(f"\nStarting PSNR and SSIM calculations for: \nAPP{bitrate}_{burst} - E2E{e2e_lat} - {sim_params} - {queues}.")
tic = time.perf_counter()


# PCAP PART: MAIN FROM PCAP-TRACES
n_ues = int(cli_args.params[-1])

seeds = cli_args.seed
seeds_to_simulate = []
seeds_to_simulate = []
for i in range(1, seeds + 1):    
    seeds_to_simulate.append(i)
print("Seeds to simulate:", seeds_to_simulate, "\n")

for seed in range(1, seeds + 1):
    
    tic_seed = time.perf_counter()

    for ue in range(n_ues):
        
        # Create modified pcap trace with PDR statistics
        pdr_file = main(cli_args, ue, n_ues, seed)

        # gstreamer - create uncompressed YUV from pcap for best accuracy
        temp_video_name = f"tempyuv_{temp_file_name}.yuv"
        gst_cmd = 'gst-launch-1.0 filesrc location="{trace}" ! pcapparse caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! filesink location="{output}"'.format(
            trace=cli_args.output, output=temp_video_name)
        result = subprocess.run(gst_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        
        os.remove(cli_args.output)
        
        # toc_test = time.perf_counter()
        # print(f'Time Elapsed: {int(toc_test-tic)} seconds.')
        
        # print(f"After YUV conversion, UE{ue}, SEED{seed}") # WORKS FOR SURE UNTIL HERE!!!
        # raise SystemExit()
        
        # ffmpeg - convert YUV to MP4 to prepare for PSNR measurement
        
        # ADD: IF BITRATE = 50: FULL HD !!!
        #      ELSE: 4K RESOLUTION !!!
        # CHECK VIDEO TIMER!!!!!
        converted_file_name = f"tempmp4_{temp_file_name}.mp4"
        if cli_args.bitrate == 50:
            convert_cmd = 'ffmpeg -s 1920x1080 -i "{input}" -ss 00:00:00 -c:v libx264 -s:v 1920x1080 "{converted}"'.format(
            input=temp_video_name,converted=converted_file_name)
        else:     
            convert_cmd = 'ffmpeg -s 3840x2160 -i "{input}" -ss 00:00:00 -c:v libx264 -s:v 3840x2160 "{converted}"'.format(
            input=temp_video_name,converted=converted_file_name)
        result = subprocess.run(convert_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        os.remove(temp_video_name)
        
                
        tocc_test = time.perf_counter()
        # print(f'Time Elapsed: {int(tocc_test-tic)} seconds.')
        print(f"Finished YUV to MP4 conversion, UE{ue}, SEED{seed}") # WORKS FOR SURE UNTIL HERE!!!
        # raise SystemExit()
        
        
        
        # ffmpeg get PSNR with shell
        #psnr_file_name='psnr.txt'
        #psnr_cmd = 'ffmpeg -i {modified} -i {original} -lavfi psnr=stats_file={psnr_logfile} -f null -'.format(modified=converted_file_name, original=cli_args.video,psnr_logfile=psnr_file_name)
        #result = subprocess.run(psnr_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        #os.remove(converted_file_name)
        
        # get PSNR with ffmpeg-quality-metrics
        
        ffqm = FfmpegQualityMetrics(converted_file_name, og_video)
        temp = ffqm.calc(["ssim", "psnr"])
        df_psnr = pd.DataFrame.from_dict(temp['psnr'])
        df_ssim = pd.DataFrame.from_dict(temp['ssim'])
        
        df = pd.DataFrame.from_dict(temp['psnr'])
        os.remove(converted_file_name)
        
        # TODO: CALCULATE AND SAVE ONLY AVERAGES (PSNR - INF!!!)
        
        mse_avg = np.mean(df_psnr["mse_avg"]) 
        psnr_avg = np.array([pdr_file, round(20 * np.log10(255) - 10 * np.log10(mse_avg), 3)])
        
        ssim_avg = np.array([pdr_file, round(np.mean(df_ssim["ssim_avg"]), 4)])       
        
        
        # TODO: SAVE TO CORRECT OUTPUT PATH
        output_folder = os.getcwd() + f"\\PSNR\\PSNR-Stats\\{temp_file_name}\\SEED{seed}\\UE{ue}\\"
        output_file_psnr = "psnr.csv"
        output_file_ssim = "ssim.csv"

        os.makedirs(output_folder, exist_ok=True)
        output_full_name_psnr = os.path.join(output_folder, output_file_psnr) 
        output_full_name_ssim = os.path.join(output_folder, output_file_ssim)     

        # saving the dataframe        
        np.savetxt(output_full_name_psnr, psnr_avg, encoding='utf-8')
        np.savetxt(output_full_name_ssim, ssim_avg, encoding='utf-8')
        
        print("Finished PSNR Calculation.") 
        toc_test = time.perf_counter()
        print(f'SEED{seed} out of {seeds} - UE{ue}, Time Elapsed: {int(toc_test-tic)} seconds.\n')


toc_end = time.perf_counter()    
print(f'Finished All {seeds} Seeds- Total Time Elapsed: {int(toc_end-tic)/60} minutes.')


