# pcap-traces

Use the following command to dump the psnr per frame into a csv file:

```bash
$ python3 process_video.py --pcap <pcap file of streamed video> --video <reference mp4 video to calculate psnr>
```

Use the following command to capture an RTP stream into a pcap file:

```bash
$ sudo tcpdump -n -i lo udp port 5000 -w input_0.pcap
```

Use the following command to generate a stream of 60 seconds:

```bash
$ timeout 60 gst-launch-1.0 filesrc location=input_0.h264 ! h264parse ! avdec_h264 ! videoscale ! video/x-raw,width=1920,height=1080 ! x264enc tune=zerolatency speed-preset=superfast key-int-max=10 bitrate=10000 ! rtph264pay config-interval=1 ! multiudpsink clients="127.0.0.1:5000"
```

To analyze (and later process) the pcap file, use the following command:

```bash
$ python -m pcap_traces --pcap input_0.pcap --output output_0.pcap > trace_0.csv
```

To replay the pcap trace or processed output, use gstreamer:

```bash
$ gst-launch-1.0 filesrc location=output_0.pcap ! pcapparse caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! aasink
```

To save the capture to mp4 file:

```bash
$ gst-launch-1.0 filesrc location=output_0.pcap ! pcapparse caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse ! mp4mux ! filesink location=output_0.mp4
```

To compare PSNR (using ffmpeg in Docker):

```bash
$ docker run --rm -ti -v $(pwd):/video jrottenberg/ffmpeg:latest -i /video/output_0.mp4 -i /video/input_0.mp4 -lavfi psnr=stats_file=/video/psnr_logfile.txt -f null -
```
