import sounddevice as sd
import queue
import numpy as np
import threading
import tkinter as tk
from tkinter import colorchooser
from faster_whisper import WhisperModel
import torch
import time
import logging
import gc
import os

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# Config
DEVICE_NAME = "CABLE Output"

MODEL_SIZE = "large-v3"

CAPTION_BG = "#000000"
CAPTION_FG = "#ffffff"
CAPTION_OPACITY = 0.85

SAMPLE_RATE = 16000
BLOCKSIZE = 1024
CHANNELS = 1

TRANSCRIBE_EVERY = 1.2
BUFFER_SECONDS = 4
ENERGY_THRESHOLD = 0.0003

# Pick GPU else CPU
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
COMPUTE_TYPE = "float16" if USE_CUDA else "int8"

print("CUDA Available:", USE_CUDA)

audio_q = queue.Queue(maxsize=20)
text_q = queue.Queue(maxsize=50)

audio_buffer = []
history = []

model = None
model_lock = threading.Lock()
switching = False

def callback(indata, frames, time_info, status):
    mono = indata.mean(axis=1).copy()

    try:
        audio_q.put_nowait(mono)
    except queue.Full:
        pass

def load_model(name):
    global model, switching, MODEL_SIZE

    MODEL_SIZE = name
    switching = True

    print("Loading model:", name)

    with model_lock:
        try:
            del model
        except:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        model = WhisperModel(
            name,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )

    switching = False

# Realtime transcription
def speech_loop():
    global model

    load_model(MODEL_SIZE)

    devices = sd.query_devices()
    device_id = None

    if DEVICE_NAME.strip():
        for i, d in enumerate(devices):
            if (
                d["max_input_channels"] > 0
                and DEVICE_NAME.lower()
                in d["name"].lower()
            ):
                device_id = i
                break

    if device_id is None:
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                device_id = i
                break

    print("Using device:", devices[device_id]["name"])

    last_transcribe = time.time()

    with sd.InputStream(
        device=device_id,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="float32",
        blocksize=BLOCKSIZE,
        callback=callback
    ):
        while True:
            chunk = audio_q.get()
            audio_buffer.extend(chunk)

            max_samples = int(
                SAMPLE_RATE * BUFFER_SECONDS
            )

            if len(audio_buffer) > max_samples:
                del audio_buffer[:-max_samples]

            now = time.time()

            if now - last_transcribe >= TRANSCRIBE_EVERY:
                last_transcribe = now

                if switching:
                    continue

                if len(audio_buffer) < SAMPLE_RATE:
                    continue

                audio_np = np.array(
                    audio_buffer,
                    dtype=np.float32
                )

                energy = np.abs(audio_np).mean()

                if energy < ENERGY_THRESHOLD:
                    continue

                try:
                    with model_lock:
                        segments, _ = model.transcribe(
                            audio_np,
                            language="en",
                            beam_size=1,
                            vad_filter=False,
                            condition_on_previous_text=False
                        )

                    text = " ".join(
                        s.text.strip()
                        for s in segments
                    ).strip()

                    if text:
                        text_q.put(text)

                except Exception as e:
                    print("Error:", e)

class HistoryWindow:
    def __init__(self):
        self.win = tk.Toplevel()
        self.win.title("Subtitle History")
        self.win.geometry("500x400")
        self.win.configure(bg="black")
        self.win.attributes("-topmost", True)

        self.text = tk.Text(
            self.win,
            bg="black",
            fg="white",
            font=("Consolas", 12)
        )

        self.text.pack(
            expand=True,
            fill="both"
        )

        self.update()

    def update(self):
        self.text.delete("1.0", tk.END)

        for line in history[-15:]:
            self.text.insert(
                tk.END,
                f">> {line}\n"
            )

        self.win.after(
            500,
            self.update
        )

class OptionsWindow:
    def __init__(self, on_model_change, overlay):
        self.on_model_change = on_model_change
        self.overlay = overlay

        self.win = tk.Toplevel()
        self.win.title("Options")
        self.win.geometry("400x430")
        self.win.configure(bg="#111")
        self.win.attributes("-topmost", True)

        tk.Label(
            self.win,
            text="Model",
            fg="white",
            bg="#111",
            font=("Segoe UI", 12, "bold")
        ).pack(pady=10)

        models = [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
            "distil-large-v3"
        ]

        self.model_var = tk.StringVar(value=MODEL_SIZE)

        tk.OptionMenu(
            self.win,
            self.model_var,
            *models
        ).pack()

        tk.Button(
            self.win,
            text="Caption Background Color",
            command=self.pick_bg
        ).pack(pady=8)

        tk.Button(
            self.win,
            text="Caption Text Color",
            command=self.pick_fg
        ).pack(pady=8)

        tk.Label(
            self.win,
            text="Opacity",
            fg="white",
            bg="#111"
        ).pack()

        self.opacity = tk.DoubleVar(
            value=CAPTION_OPACITY
        )

        tk.Scale(
            self.win,
            from_=0.2,
            to=1.0,
            resolution=0.05,
            orient="horizontal",
            variable=self.opacity,
            command=self.change_opacity,
            length=250
        ).pack()

        tk.Button(
            self.win,
            text="Apply",
            command=self.apply
        ).pack(pady=15)

    def pick_bg(self):
        global CAPTION_BG
        c = colorchooser.askcolor(CAPTION_BG)[1]
        if c:
            CAPTION_BG = c
            self.overlay.update_style()

    def pick_fg(self):
        global CAPTION_FG
        c = colorchooser.askcolor(CAPTION_FG)[1]
        if c:
            CAPTION_FG = c
            self.overlay.update_style()

    def change_opacity(self, value):
        global CAPTION_OPACITY
        CAPTION_OPACITY = float(value)
        self.overlay.update_style()

    def apply(self):
        self.on_model_change(
            self.model_var.get()
        )

class CaptionOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Live Captions")

        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)

        w, h = 1000, 120

        x = (
            self.root.winfo_screenwidth() - w
        ) // 2

        y = (
            self.root.winfo_screenheight()
            - h
            - 80
        )

        self.root.geometry(
            f"{w}x{h}+{x}+{y}"
        )

        self.label = tk.Label(
            self.root,
            text="Listening...",
            font=("Segoe UI", 20),
            wraplength=950,
            justify="center",
            anchor="center",
            bg=CAPTION_BG,
            fg=CAPTION_FG
        )

        self.label.pack(
            expand=True,
            fill="both",
            padx=12,
            pady=12
        )

        self.offset_x = 0
        self.offset_y = 0

        self.label.bind("<Button-1>", self.start_move)
        self.label.bind("<B1-Motion>", self.do_move)

        self.root.bind("<Button-1>", self.start_move)
        self.root.bind("<B1-Motion>", self.do_move)

        self.start_width = 0
        self.start_height = 0
        self.start_x = 0
        self.start_y = 0

        self.resize_grip = tk.Label(
            self.root,
            text="◢",
            font=("Segoe UI", 12),
            bg=CAPTION_BG,
            fg=CAPTION_FG,
            cursor="bottom_right_corner"
        )

        self.resize_grip.place(
            relx=1.0,
            rely=1.0,
            anchor="se"
        )

        self.resize_grip.bind(
            "<Button-1>",
            self.start_resize
        )

        self.resize_grip.bind(
            "<B1-Motion>",
            self.do_resize
        )

        self.update_style()

        tk.Button(
            self.root,
            text="📋",
            command=self.open_history
        ).place(x=10, y=10)

        tk.Button(
            self.root,
            text="⚙",
            command=self.open_options
        ).place(x=50, y=10)

        tk.Button(
            self.root,
            text="✖",
            command=self.quit_app,
            bg="darkred",
            fg="white"
        ).place(x=90, y=10)

        self.history_win = None
        self.options_win = None
        self.latest = ""

        self.update_loop()

    def start_move(self, event):
        self.offset_x = event.x_root - self.root.winfo_x()
        self.offset_y = event.y_root - self.root.winfo_y()

    def do_move(self, event):
        x = event.x_root - self.offset_x
        y = event.y_root - self.offset_y
        self.root.geometry(f"+{x}+{y}")

    def start_resize(self, event):
        self.start_width = self.root.winfo_width()
        self.start_height = self.root.winfo_height()
        self.start_x = event.x_root
        self.start_y = event.y_root

    def do_resize(self, event):
        dx = event.x_root - self.start_x
        dy = event.y_root - self.start_y

        width = max(300, self.start_width + dx)
        height = max(60, self.start_height + dy)

        x = self.root.winfo_x()
        y = self.root.winfo_y()

        self.root.geometry(
            f"{width}x{height}+{x}+{y}"
        )

        self.label.configure(
            wraplength=max(200, width - 40)
        )

        self.resize_grip.place(
            relx=1.0,
            rely=1.0,
            anchor="se"
        )

    def update_style(self):
        self.root.configure(bg=CAPTION_BG)

        self.label.configure(
            bg=CAPTION_BG,
            fg=CAPTION_FG
        )

        self.resize_grip.configure(
            bg=CAPTION_BG,
            fg=CAPTION_FG
        )

        self.root.attributes(
            "-alpha",
            CAPTION_OPACITY
        )

    def quit_app(self):
        os._exit(0)

    def open_history(self):
        if (
            self.history_win is None
            or not self.history_win.win.winfo_exists()
        ):
            self.history_win = HistoryWindow()

    def open_options(self):
        if (
            self.options_win is None
            or not self.options_win.win.winfo_exists()
        ):
            self.options_win = OptionsWindow(
                self.change_model,
                self
            )

    def change_model(self, name):
        threading.Thread(
            target=load_model,
            args=(name,),
            daemon=True
        ).start()

    def update_loop(self):
        global history

        try:
            while True:
                self.latest = text_q.get_nowait()
                history.append(self.latest)

                if len(history) > 15:
                    history = history[-15:]

        except queue.Empty:
            pass

        self.label.config(
            text=self.latest or "Listening..."
        )

        self.root.after(
            50,
            self.update_loop
        )

    def run(self):
        self.root.mainloop()

# Init
if __name__ == "__main__":
    threading.Thread(
        target=speech_loop,
        daemon=True
    ).start()

    CaptionOverlay().run()