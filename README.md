# Explano

Convert an online lecture video (.mp4) into a pdf with the transcript per corresponding slide per page. The audio2text convertion is performed by the **tiny.en [faster-whisper] model**.

### Usage

Python version `3.11.0` and Torch with `+cu128` is used to develop this pipeline. Select an interpreter and install the required modules. Initially, paste an .mp4 file of an online video lecture which contains the slides somewhere in the video frame with a constant aspect ratio. Run the pipeline by `python main.py`. The following shows a repository example. Insert the video into the **input** folder.

```
Explano
│   README.md
│   main.py
│   requirements.txt
└───intput
│   │   Data_Mining_lec1.mp4
│   │   (Data_Mining_lec1.mp3)
└───slides
│   │   (slide_000_0.0s.jpg)
└───output
    │   (Data_Mining_lec1.pdf)
```

### Pipeline

<ul>
  <li>Setup: initialize the necessary directories.</li>
  <li>Select Region of Interest: manually select the area where the slide is displayed.</li>
  <li>Detect transitions: detect the timesteps when slide transitions most likely occur based on the structural similarity index (SSIM).</li>
  <li>Setup Whisper: load the selected faster-whisper model.</li>
  <li>Transcribe audio: obtain the index, text and timestamps for each extracted text segment from the audio file.</li>
  <li>Transcript per slide: sort the text segments within the timespan of each shown slide and combine the text segmnts into a cleaned-up transcript per slide.</li>
  <li>Plot histograms: of (1) the number of segments per slide over time (changing bar widths), and (2) the number of segments per slide index. </li>
  <li>Create PDF: create a PDF where each page contains a distinct slide image and its corresponding transcription.</li>
</ul>
