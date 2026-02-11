import os
import subprocess
import cv2
import torch
import time
import re
import matplotlib.pyplot as plt
from decord import VideoReader, cpu
from fpdf import FPDF
from faster_whisper import WhisperModel
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create directories and extract audio from video
def setup(input_path, slides_path, output_path, video_path, audio_path):

    directories = [input_path, slides_path, output_path]

    # create directories if non existing
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # concatenate multiple video parts by e.g.:
    # ffmpeg -i 12_MAIR_part1.mp4 -i 12_MAIR_part2.mp4 -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]" -map "[v]" -map "[a]" 12_MAIR_lec.mp4

    # remove the files in the slides folder before each new run
    if os.path.exists(slides_path):
        for file in os.listdir(slides_path):
            file_path = os.path.join(slides_path,file) 
            if os.path.isfile(file_path):
                os.unlink(file_path)

    # extract audio from video if not already done
    if not os.path.exists(audio_path):

        # in the command line extract audio with ffmpeg:
        # have an input i, ignore the video extraction by video none (-vn) and best audio quality (-q0)
        cmd = ["ffmpeg","-i",video_path , "-vn", "-q:a","0", audio_path]
        subprocess.run(cmd, check=True)
        print("audio extracted")

# manually select the region of the video (slide area) for which the proccessing needs to happen
def select_roi(vr, video_path, frame_idx=2000, preview_width=960):

    # extract a single frame (at index frame_idx) for the interface of the ROI to know what roi to select
    frame = vr[frame_idx].asnumpy()[:, :, ::-1] 

    # adjust the size of the extracted frame and make smaller
    H, W = frame.shape[:2]
    scale = preview_width / W
    preview_height = int(H * scale)
    preview = cv2.resize(frame, (preview_width, preview_height))

    print("select ROI...")
    # the preview pop up window allows the user to drag and select the area of the slide that needs to be processed
    roi_preview = cv2.selectROI("Select slide area (on preview)",preview,fromCenter=False,showCrosshair=True)
    cv2.destroyAllWindows()

    # rescale the selected roi back to the original video sizes
    x, y, w, h = map(float, roi_preview)
    x_full, y_full = int(x / scale), int(y / scale)
    w_full, h_full = int(w / scale), int(h / scale)

    roi = (x_full, y_full, w_full, h_full)

    print(f"ROI selected")
    return roi

def transition_detection(vr,slides_path,frame_sr=1.0,change_threshold=0.95,roi=None,resize_gray=(320, 180)):

    # extract relevant metadata from the video
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    # calculate after how many frames a new frame should be sampled 
    frame_interval = int(fps * frame_sr)

    # calculate the indexes which frames from the video are going to be sampled
    sample_indices = np.arange(0, total_frames, frame_interval)

    print(f"number of frames sampled: {len(sample_indices)}")

    x, y, w, h = roi

    print("sampling frames and converting to grayscale...")
    # extract the sampled frames and transform them into grayscale 
    all_gray = []
    for idx in sample_indices:
        frame = vr[idx].asnumpy()
        frame = frame[y:y+h, x:x+w] 
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if resize_gray:
            gray = cv2.resize(gray, resize_gray)
        all_gray.append(gray)

    print(f"frames sampled and converted to grayscale")
    print(f"calculate transitions...")

    # calculate the structural similarity index (SSIM) between adjacent frames over time
    sims = []
    for i in range(1, len(all_gray)):
        sim = ssim(all_gray[i - 1], all_gray[i])
        sims.append(sim)

    # detect a transition where the similarity drops under the change threshold
    transition_timestamps = [(0, 0)]
    for i, sim in enumerate(sims):
        if sim < change_threshold:
            timestamp = i * frame_interval / fps
            transition_timestamps.append((i + 1, timestamp))

    print(f"number of detected transtions: {len(transition_timestamps)}")
    print("exracting and saving slides...")

    # extract the frames at the transition timestamps and save them as images in the slides folder
    for frame_idx, t in transition_timestamps:
        idx = sample_indices[frame_idx]
        frame = vr[idx].asnumpy()
        frame = frame[y:y+h, x:x+w]
        out_path = os.path.join(slides_path, f"slide_{frame_idx:03d}_{t:.1f}s.jpg")
        Image.fromarray(frame).save(out_path, quality=90)

    print("slides extracted and saved")

    return transition_timestamps

def setup_whisper(model_id="large-v3", device="cuda", compute_type="float16"):

    # load the faster-whisper model
    # compute type can be "float16", "int8" (current), or "int8_float16"
    model = WhisperModel(model_id,device=device, compute_type=compute_type)  
    
    return model

def transcribe_audio(model, audio_path):
 
    start = time.time()

    print("transcribing audio...")

    # transcribe the audio into text segments 
    segments, info = model.transcribe(audio_path,vad_filter=True, beam_size=2, language="en")

    elapsed = time.time() - start

    print(f"transcription time: {elapsed:.2f} seconds")
    print("creating a segments info list...")

    # create a list of dictionaries storing information for each segment
    segments_info = []
    for idx, segment in enumerate(segments):
        segment_info = {
            "index": idx,
            "text": segment.text.strip(),
            "timestamp": [segment.start, segment.end],
        }
        segments_info.append(segment_info)

    print(f"number of segments: {len(segments_info)}")
    print(f"segmenent range over: {segments_info[0]['timestamp'][0]:.1f}s to {segments_info[-1]['timestamp'][1]:.1f}s")

    return segments_info

def transcript_per_slide(transition_timestamps, segments_info):

    print("combining transcript segments per slide...")
    
    slides_data = []

    # for each slide determine the end time 
    for i, (frame_idx, start_t) in enumerate(transition_timestamps):
        if i < len(transition_timestamps) - 1:
            end_t = transition_timestamps[i + 1][1]
        else:
            end_t = float("inf")

        # Collect transcript segments for this slide within the time range of the displayed slide
        slide_segments = []
        for seg in segments_info:
            if start_t <= seg["timestamp"][0] < end_t:
                slide_segments.append(seg)
        
        # combine the segments into a complete text for the slide
        transcripts = []
        for seg in slide_segments:
            text = seg["text"].strip()
            if text:
                transcripts.append(text)

        # finalize the transcript by removing duplicates and non-ascii characters
        paragraph = " ".join(transcripts)
        paragraph = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', paragraph, flags=re.IGNORECASE)  
        paragraph = re.sub(r'\b(\w{4,})\1+\b', r'\1', paragraph, flags=re.IGNORECASE) 
        paragraph = paragraph.encode("ascii", "replace").decode("ascii")

        slides_data.append({
            "slide_index": i,
            "frame_index": frame_idx,
            "start_time": start_t,
            "end_time": end_t,
            "segments": slide_segments,
            "num_segments": len(slide_segments),
            "transcript": paragraph,
        })

    print("all transcripts per slide are combined")

    return slides_data

def time_based_histogram(slides_data):

    # extract data per slide
    starts = [s["start_time"] for s in slides_data]
    ends = [s["end_time"] for s in slides_data]
    counts = [s["num_segments"] for s in slides_data]

    # caclulate the duration of each slide to change the bar widths in the histogram
    durations = [end - start for start, end in zip(starts, ends)]

    # plot a histogram showing the number of segments per slide over time
    # the width of the bars corresponds to the duration of each slide
    plt.figure(figsize=(12, 5))
    plt.bar(starts, counts, width=durations, align="edge",color="skyblue", edgecolor="black")
    plt.title("Segments per slide over time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of segments")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

def segment_histogram(slides_data):

    # extract the slide index and number of segments per slide
    slide_labels = []
    segment_counts = []
    for slide in slides_data:
        slide_labels.append(slide['slide_index'])
        segment_counts.append(slide["num_segments"])
 
    # plot a histogram showing the number of segments per slide
    plt.figure(figsize=(10, 4))
    plt.bar(slide_labels, segment_counts, color="skyblue", edgecolor="black")
    plt.title("Segments per slide")
    plt.xlabel("Slide Index")
    plt.ylabel("Number of segments")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

def create_pdf(slides_data, filename, slides_path, output_path, text_width = 130, text_lowering_param = 10):
    
    # setup page sizes and create a new PDF document
    page_width, page_height = 210, 297
    margin = 10
    text_line_height = 7
    pdf = FPDF(orientation="P", unit="mm", format="A4")

    # for each slide get the slide image and corresponding transcript and add a new page to the PDF for it
    for slide in slides_data:

        frame_idx = slide['frame_index']
        timestamp = slide['start_time']
        slide_name = f"slide_{frame_idx:03d}_{timestamp:.1f}s.jpg"
        slide_path = os.path.join(slides_path, slide_name)
        
        if not os.path.exists(slide_path):
            print(f"slide isng slide: {slide_path}")
            
        text = slide.get("transcript", "[No transcript available]")

        pdf.add_page()

        # open the slide image and add it to the bottom half of the portrait page
        y = margin
        if os.path.exists(slide_path):
            with Image.open(slide_path) as img:
                w, h = img.size
                aspect = w / h

                max_img_w = page_width - 2 * margin
                max_img_h = page_height / 2  # limit to half page height

                # resize the image until its height fits into the top half of the page while keeping its aspect ratio
                img_w = max_img_w
                img_h = img_w / aspect
                if img_h > max_img_h:
                    img_h = max_img_h
                    img_w = img_h * aspect 

                x = (page_width - img_w) / 2
                pdf.image(slide_path, x=x, y=y, w=img_w, h=img_h)
                y += img_h + text_lowering_param
        else:
            # if the slide image is missing, add a text that its image is missing
            pdf.set_xy(margin, y)
            pdf.set_font("Helvetica", "I", 11)
            pdf.multi_cell(text_width, 10, "Image missing", align="C")
            y += text_lowering_param
        
        # add the transcript text to the bottom half of the page, starting after the image
        pdf.set_xy(margin, y)  
        pdf.set_font("Helvetica", size=11)

        # change the width of the text box and place it to the left of the page, allowing annotations on the right
        pdf.multi_cell(text_width, text_line_height, text, align="J")

    # save the PDF to the output folder 
    print("filename", filename)
    output_file = os.path.join(output_path, f"{filename}.pdf")
    print("output file: ", output_file)
    pdf.output(output_file)
    print(f"PDF saved with name: {output_file}")

if __name__ == "__main__":

    start = time.time()

    input_path = "input"
    slides_path = "slides" 
    output_path = "output"

    filename = "4_MAIR_lec"

    video_path = os.path.join(input_path, f"{filename}.mp4")
    audio_path = os.path.join(input_path, f"{filename}.mp3")

    # setup the directories and possibly extract the audio from the video 
    setup(input_path, slides_path, output_path, video_path, audio_path)

    # load the video
    vr = VideoReader(video_path)

    # select the ROI of the slide area
    roi = select_roi(vr,video_path)

    frame_sr = 15.0
    batch_size = 32

    # change the treshold size based on the lecture theme which can change per course
    change_threshold= 0.99 # MA
    # change_threshold= 0.98 # DM

    # detect the transitions between slides based on similarity between adjacent frames
    transition_timestamps = transition_detection(vr,slides_path,frame_sr, change_threshold=0.95,roi=roi)

    # use the tiny faster-whisper model 
    model_id = "tiny.en"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype =  torch.float16 if torch.cuda.is_available() else torch.float32

    compute_type = "float32" if torch.cuda.is_available() else "int8"

    print("set up the audio2text model...")

    # load the faster-whisper model 
    pipe = setup_whisper(model_id,device,compute_type)

    print("model set up")

    # transcribe the audio and get a list of segments with info regarding its index, text and timestamps
    segments_info = transcribe_audio(pipe, audio_path)

    # get the full transcript per slide and additional metadata
    slides_data = transcript_per_slide(transition_timestamps, segments_info)

    # analyse the disribution of text per slide and the lenght of the slides displayed
    time_based_histogram(slides_data)
    segment_histogram(slides_data)

    # create a PDF with the slide and its corresponding transcript 
    create_pdf = create_pdf(slides_data, filename, slides_path, output_path)

    elapsed = time.time() - start

    print(f"total processing time: {elapsed:.2f} seconds")





