# live-caption-system
Real-Time Live Captioning with Faster-Whisper
----------------------------------------------
A Python-based real-time live captioning tool that transcribes system or microphone audio into text using the faster-whisper implementation of OpenAI’s Whisper model. The application features a Tkinter-based GUI for controls and display, supports selectable audio input devices, and automatically detects GPU availability to enable CUDA acceleration for improved performance and lower latency.

Users can choose between different Whisper model sizes (default: large), and customize the caption display including text color, background color, and background opacity. Built with torch, sounddevice, numpy, and threading to handle real-time audio processing and transcription efficiently.
