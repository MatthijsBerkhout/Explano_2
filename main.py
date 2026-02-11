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

def setup(input_path, slides_path, output_path, video_path, audio_path):

    directories = [input_path, slides_path, output_path]

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

    if not os.path.exists(audio_path):
        # extract_audio(audio_path)
        cmd = ["ffmpeg","-i",video_path , "-vn", "-q:a","0", audio_path]
        # cmd = ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
        subprocess.run(cmd, check=True)
        print("Extract audio")

def select_roi(video_path, frame_idx=2000, preview_width=960):

    vr = VideoReader(video_path)
    frame = vr[frame_idx].asnumpy()[:, :, ::-1]  # BGR for OpenCV
    H, W = frame.shape[:2]

    # ---- Resize frame for comfortable viewing ----
    scale = preview_width / W
    preview_height = int(H * scale)
    preview = cv2.resize(frame, (preview_width, preview_height))

    # ---- Let user draw ROI on the smaller preview ----
    roi_preview = cv2.selectROI(
        "Select slide area (on preview)",
        preview,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyAllWindows()

    # ---- Rescale ROI coordinates to original resolution ----
    x, y, w, h = map(float, roi_preview)
    x_full, y_full = int(x / scale), int(y / scale)
    w_full, h_full = int(w / scale), int(h / scale)

    roi = (x_full, y_full, w_full, h_full)
    print(f"📦 ROI selected (original scale): {roi}")
    return roi

def extract_slides_ssim(
    video_path,
    slides_path,
    frame_sr=1.0,
    change_threshold=0.95,
    roi=None,
    resize_gray=(320, 180),
):
    """
    Fast slide transition detection using SSIM on grayscale frames.
    Keeps colored output without holding all color frames in memory.
    """
    os.makedirs(slides_path, exist_ok=True)

    # ---- Load video ----
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    frame_interval = int(fps * frame_sr)
    sample_indices = np.arange(0, total_frames, frame_interval)

    print(f"🎞 Sampling every {frame_sr}s ({len(sample_indices)} frames)")
    if roi is not None:
        x, y, w, h = map(int, roi)
        print(f"🎯 Using ROI: x={x}, y={y}, w={w}, h={h}")

    # ---- Sample only grayscale frames ----
    all_gray = []
    for idx in sample_indices:
        frame = vr[idx].asnumpy()
        if roi is not None:
            frame = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if resize_gray:
            gray = cv2.resize(gray, resize_gray)
        all_gray.append(gray)

    print(f"🖼️ Sampled {len(all_gray)} grayscale frames. Computing SSIM...")

    # ---- Compute SSIM between consecutive frames ----
    sims = [ssim(all_gray[i - 1], all_gray[i]) for i in range(1, len(all_gray))]

    # ---- Detect transitions ----
    transition_timestamps = [(0, 0)]
    for i, s in enumerate(sims):
        if s < change_threshold:
            timestamp = i * frame_interval / fps
            transition_timestamps.append((i + 1, timestamp))

    print(f"⚡ Detected {len(transition_timestamps)} transitions")

    # ---- Re-open color frames only for transition indices ----
    for frame_idx, t in transition_timestamps:
        idx = sample_indices[frame_idx]
        frame = vr[idx].asnumpy()
        if roi is not None:
            print(frame.shape)
            frame = frame[y:y+h, x:x+w]
        print(frame.shape)
        out_path = os.path.join(slides_path, f"slide_{frame_idx:03d}_{t:.1f}s.jpg")
        Image.fromarray(frame).save(out_path, quality=90)
        # print(f"🖼️ Saved slide at {t:.1f}s")

    print("✅ Extraction complete.")
    return transition_timestamps

def setup_whisper(model_id="large-v3", device="cuda", compute_type="float16"):
    """
    Load a Faster-Whisper model efficiently.
    """
    print(f"🧩 Loading Faster-Whisper model ({model_id}) on {device}...")

    # Load the optimized Faster-Whisper model
    model = WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,  # "float16", "int8", or "int8_float16"
    )

    return model

def transcribe_audio(model, audio_path):
    """
    Transcribe audio using the Faster-Whisper model with built-in VAD.
    """
    print(f"🎧 Transcribing {audio_path} ...")
    start = time.time()

    # Transcribe with built-in silence removal (VAD)
    segments, info = model.transcribe(
        audio_path,
        vad_filter=True,  # 🔥 skip silence automatically
        # vad_parameters=dict(
        #     min_speech_duration_ms=250,
        #     silence_duration_ms=500,
        #     speech_pad_ms=150,
        # ),
        beam_size=2,         # small beam for balance between speed/accuracy
        # temperature=0.0,     # deterministic decoding
        language="en",       # optional (skip detection)
    )

    elapsed = time.time() - start
    print(f"⏱️ Transcription completed in {elapsed:.2f}s")
    print(f"🌐 Detected language: {info.language}")

    # # Convert segments to list of dicts for compatibility with your previous logic
    indexed_segments = [
        {
            "index": idx,
            "text": segment.text.strip(),
            "timestamp": [segment.start, segment.end],
        }
        for idx, segment in enumerate(segments)
    ]

    print(f"🗒️ Segments: {len(indexed_segments)} | Coverage: {indexed_segments[0]['timestamp'][0]:.1f}s → {indexed_segments[-1]['timestamp'][1]:.1f}s")

    return indexed_segments

def transcript_per_slide(transition_timestamps, indexed_segments,device):

    slides_data = []

    for i, (frame_idx, start_t) in enumerate(transition_timestamps):
        # Determine slide end (next start or end of audio)
        if i < len(transition_timestamps) - 1:
            end_t = transition_timestamps[i + 1][1]
        else:
            end_t = float("inf")

        # Collect transcript segments for this slide
        slide_segments = []
        for seg in indexed_segments:
            if start_t <= seg["timestamp"][0] < end_t:
                slide_segments.append(seg)
        
        transcripts = []
        for seg in slide_segments:
            text = seg["text"].strip()
            if text:
                transcripts.append(text)

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

    return slides_data

def time_based_histogram(slides_data):
    # Extract data
    starts = [s["start_time"] for s in slides_data]
    ends = [s["end_time"] for s in slides_data]
    counts = [s["num_segments"] for s in slides_data]
    labels = [f"Slide {s['slide_index']:02d}" for s in slides_data]

    durations = [end - start for start, end in zip(starts, ends)]

    plt.figure(figsize=(12, 5))
    plt.bar(starts, counts, width=durations, align="edge",
            color="skyblue", edgecolor="black")

    plt.title("Speech Segments per Slide (Timeline)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of Speech Segments")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

def segment_histogram(slides_data):
    slide_labels = [f"{s['slide_index']:02d}" for s in slides_data]
    segment_counts = [s["num_segments"] for s in slides_data]
    # --- Plot histogram ---
    plt.figure(figsize=(10, 4))
    plt.bar(slide_labels, segment_counts, color="skyblue", edgecolor="black")
    plt.title("Speech Segments per Slide")
    plt.xlabel("Slide Index")
    plt.ylabel("Number of Speech Segments")
    plt.tight_layout()
    plt.show()

def create_pdf(slides_data, filename, slides_path, output_path, text_width = 130, text_lowering_param = 10):
    page_width, page_height = 210, 297   # A4 in mm
    margin = 10
    text_line_height = 7

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    os.makedirs(output_path, exist_ok=True)

    for slide in slides_data:
        # Fix: Match the slide naming format from extract_slides function
        frame_idx = slide['frame_index']
        timestamp = slide['start_time']
        slide_name = f"slide_{frame_idx:03d}_{timestamp:.1f}s.jpg"
        slide_path = os.path.join(slides_path, slide_name)
        
        if not os.path.exists(slide_path):
            print(f"⚠️ Missing slide: {slide_path}")
            
        text = slide.get("transcript", "[No transcript available]")

        pdf.add_page()

        # --- IMAGE ---
        y = margin
        if os.path.exists(slide_path):
            with Image.open(slide_path) as img:
                w, h = img.size
                aspect = w / h

                max_img_w = page_width - 2 * margin
                max_img_h = page_height / 2  # limit to half page height

                # scale respecting both limits
                img_w = max_img_w
                img_h = img_w / aspect
                if img_h > max_img_h:
                    img_h = max_img_h
                    img_w = img_h * aspect 

                x = (page_width - img_w) / 2
                pdf.image(slide_path, x=x, y=y, w=img_w, h=img_h)
                y += img_h + text_lowering_param
        else:
            pdf.set_xy(margin, y)
            pdf.set_font("Helvetica", "I", 11)
            pdf.multi_cell(text_width, 10, "[Image missing]", align="C")
            y += text_lowering_param
        
        # --- TEXT --- #
        pdf.set_xy(margin, y)   # << fixed Y starting point
        pdf.set_font("Helvetica", size=11)
        # text_end_y = pdf.get_y()
        pdf.multi_cell(text_width, text_line_height, text, align="J")
        # text_height = pdf.get_y() - text_end_y

        # Draw vertical line after text
        # line_x = margin + text_width
        # pdf.set_draw_color(100, 100, 100)  # Black color
        # pdf.set_line_width(0.3)      # Line thickness in mm
        # pdf.line(line_x, y, line_x, y + text_height)

    # --- SAVE PDF ---
    print("filename", filename)
    output_file = os.path.join(output_path, f"{filename}.pdf")
    print("output file: ", output_file)
    pdf.output(output_file)
    print(f"✅ Saved PDF with {len(slides_data)} slides → {output_file}")




if __name__ == "__main__":

    input_path = "input"
    slides_path = "slides" 
    output_path = "output"

    filename = "4_MAIR_lec"

    video_path = os.path.join(input_path, f"{filename}.mp4")
    audio_path = os.path.join(input_path, f"{filename}.mp3")

    setup(input_path, slides_path, output_path, video_path, audio_path)

    roi = select_roi(video_path)

    frame_sr = 15.0
    batch_size = 32
    change_threshold= 0.99 # MA
    # change_threshold= 0.98 # DM

    transition_timestamps = extract_slides_ssim(
        video_path,
        slides_path,
        frame_sr=frame_sr,          # every second
        change_threshold=0.95,
        roi = roi,
    )

    model_id = "tiny.en"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype =  torch.float16 if torch.cuda.is_available() else torch.float32

    compute_type = "float32" if torch.cuda.is_available() else "int8"

    pipe = setup_whisper(
        model_id=model_id,
        device=device,
        compute_type=compute_type
    )

    indexed_segments = transcribe_audio(pipe, audio_path)

    slides_data = transcript_per_slide(transition_timestamps, indexed_segments,device)
    time_based_histogram(slides_data)
    segment_histogram(slides_data)

    create_pdf = create_pdf(slides_data, filename, slides_path, output_path)





