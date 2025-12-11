# main.py - VERSIÓN FINAL 100% FUNCIONAL - BOTONES PEQUEÑOS Y TODO FUNCIONA
import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import pyaudio
import wave
from pydub import AudioSegment
from pydub.playback import play
import threading
import os
import time

# KERNELS
kernels = {
    'Sobel Ambos': (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])),
    'Laplace': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
}

class SignalProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Señales Multimedia")
        self.root.geometry("1000x700")
        self.root.configure(bg="#0d1117")
        self.root.resizable(False, False)

        self.photo = None
        self.audio = None
        self.audio_rate = None
        self.audio_path = None
        self.video_frames = []
        self.video_path = None

        # Header
        header = tk.Frame(root, bg="#161b22", height=120)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="PROCESADOR DE SEÑALES", fg="#58a6ff", bg="#161b22", 
                font=("Segoe UI", 24, "bold")).pack(pady=20)
        tk.Label(header, text="Proyecto Integrador", fg="#8b949e", bg="#161b22").pack()

        self.container = tk.Frame(root, bg="#0d1117")
        self.container.pack(fill="both", expand=True, padx=100, pady=30)
        self.show_main_menu()

    def clear(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    def show_main_menu(self):
        self.clear()
        tk.Label(self.container, text="Selecciona el tipo de señal", 
                fg="#58a6ff", bg="#0d1117", font=("Segoe UI", 22, "bold")).pack(pady=40)

        buttons = [("Fotografía", "#ff7b72"), ("Clip de Audio", "#ffa657"), ("Clip de Video", "#79c0ff")]
        for text, color in buttons:
            btn = tk.Button(self.container, text=text, bg=color, fg="white",
                           font=("Segoe UI", 15, "bold"), width=20, height=2,
                           relief="flat", bd=0, cursor="hand2",
                           command=lambda t=text: self.section_menu(t))
            btn.pack(pady=18)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg="white", fg=color))
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c, fg="white"))

    def section_menu(self, title):
        self.clear()
        tk.Button(self.container, text="Volver", bg="#30363d", fg="#58a6ff",
                 font=("Segoe UI", 10, "bold"), command=self.show_main_menu).pack(anchor="w", padx=20, pady=10)
        tk.Label(self.container, text=title, fg="#58a6ff", bg="#0d1117",
                font=("Segoe UI", 20, "bold")).pack(pady=25)

        if "Foto" in title:
            self.photo_menu()
        elif "Audio" in title:
            self.audio_menu()
        else:
            self.video_menu()

    def create_btn(self, text, cmd, color="#238636"):
        btn = tk.Button(self.container, text=text, bg=color, fg="white",
                       font=("Segoe UI", 11, "bold"), width=38, height=1,
                       relief="flat", bd=0, cursor="hand2", command=cmd,
                       padx=20, pady=10)
        btn.pack(pady=6)
        btn.bind("<Enter>", lambda e, b=btn: b.config(bg="white", fg=color))
        btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c, fg="white"))

    def photo_menu(self):
        self.create_btn("Capturar Foto", self.capture_photo, "#238636")
        self.create_btn("Cargar Imagen", self.upload_photo, "#8957e5")
        self.create_btn("Mostrar Original", lambda: self.display_image(self.photo, "Original"), "#1f6feb")
        self.create_btn("Filtro Sobel", lambda: self.apply_filter("Sobel Ambos"), "#f85149")
        self.create_btn("Filtro Laplace", lambda: self.apply_filter("Laplace"), "#d29922")
        self.create_btn("Downsampling", self.apply_sampling_image, "#39c5bb")

    def audio_menu(self):
        self.create_btn("Grabar Voz (5s)", self.capture_audio, "#238636")
        self.create_btn("Cargar Audio", self.upload_audio, "#8957e5")
        self.create_btn("Convolución + Reverb (Fig. 13-16)", self.convolve_audio, "#ff7b72")
        self.create_btn("Muestreo Audio (Fig. 17)", self.apply_sampling_audio, "#f85149")

    def video_menu(self):
        self.create_btn("Grabar Video", self.capture_video, "#238636")
        self.create_btn("Cargar Video", self.upload_video, "#8957e5")
        self.create_btn("Reproducir Video", self.play_video, "#1f6feb")
        self.create_btn("Filtro Laplace (Fig. 20)", self.apply_laplace_to_video, "#ff7b72")
        self.create_btn("Muestreo Video (Fig. 21)", self.apply_sampling_video, "#f85149")

    # =================== CAPTURA Y CARGA ===================
    def capture_photo(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.photo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            messagebox.showinfo("Éxito", "Foto capturada")
        else:
            messagebox.showerror("Error", "No se pudo capturar")

    def upload_photo(self):
        path = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            img = cv2.imread(path, 0)
            if img is not None:
                self.photo = img
                messagebox.showinfo("Éxito", "Imagen cargada")
            else:
                messagebox.showerror("Error", "No se pudo cargar la imagen")

    def capture_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        frames = []
        messagebox.showinfo("Grabando", "Habla ahora - 5 segundos")
        for _ in range(215):
            data = stream.read(1024)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

        self.audio_path = "voz_grabada.wav"
        wf = wave.open(self.audio_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.audio_rate, self.audio = wavfile.read(self.audio_path)
        messagebox.showinfo("Éxito", "Audio grabado")

    def upload_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
        if path:
            if path.endswith('.mp3'):
                AudioSegment.from_mp3(path).export("temp.wav", format="wav")
                path = "temp.wav"
            self.audio_path = path
            self.audio_rate, self.audio = wavfile.read(path)
            messagebox.showinfo("Éxito", "Audio cargado")

    def capture_video(self):
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_path = "video_grabado.avi"
        out = cv2.VideoWriter(self.video_path, fourcc, 20.0, (640, 480))
        self.video_frames = []
        messagebox.showinfo("Grabando", "Grabando 5 segundos...")
        for _ in range(100):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                self.video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        out.release()
        messagebox.showinfo("Éxito", "Video grabado")

    def upload_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")])
        if path:
            cap = cv2.VideoCapture(path)
            self.video_frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                self.video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cap.release()
            self.video_path = path
            messagebox.showinfo("Éxito", f"Video cargado ({len(self.video_frames)} frames)")

    # =================== VISUALIZACIÓN Y FIGURAS ===================
    def display_image(self, img, title=""):
        if img is None: 
            messagebox.showerror("Error", "No hay imagen")
            return
        plt.figure(figsize=(10,8), facecolor='#0d1117')
        plt.imshow(img, cmap='gray')
        plt.title(title, color='white', fontsize=16)
        plt.axis('off')
        plt.show()

    def apply_filter(self, name):
        if self.photo is None: 
            messagebox.showerror("Error", "Carga una imagen primero")
            return
        k = kernels[name]
        if name == 'Sobel Ambos':
            gx = signal.convolve2d(self.photo, k[0], mode='same')
            gy = signal.convolve2d(self.photo, k[1], mode='same')
            filtered = np.sqrt(gx**2 + gy**2)
        else:
            filtered = signal.convolve2d(self.photo, k, mode='same')
        self.display_image(np.abs(filtered), f"Filtro {name}")

    def apply_sampling_image(self):
        if self.photo is None: return
        down = self.photo[::4, ::4]
        plt.figure(figsize=(14,7), facecolor='#0d1117')
        plt.subplot(121); plt.imshow(self.photo, cmap='gray'); plt.title("Original", color='white'); plt.axis('off')
        plt.subplot(122); plt.imshow(down, cmap='gray'); plt.title("Downsampling x4", color='white'); plt.axis('off')
        plt.suptitle("Muestreo en Imagen", color='#58a6ff', fontsize=16)
        plt.show()

    def convolve_audio(self):
        if self.audio is None: 
            messagebox.showerror("Error", "Primero graba o carga tu voz")
            return
        path = filedialog.askopenfilename(title="Selecciona impulso de reverb", filetypes=[("WAV","*.wav")])
        if not path: return
        rate2, audio2 = wavfile.read(path)
        conv = signal.convolve(self.audio, audio2, mode='full')
        conv = np.int16(conv / np.max(np.abs(conv)) * 32767)
        wavfile.write("reverb.wav", 44100, conv)

        def plot(title, data, color, file):
            plt.figure(figsize=(10,5), facecolor='#0d1117')
            f = np.fft.fftfreq(len(data), 1/44100)
            s = np.abs(np.fft.fft(data))
            plt.plot(f[:len(f)//2], s[:len(s)//2], color=color)
            plt.title(title, color='white')
            plt.xlim(0,8000)
            plt.gca().set_facecolor('#161b22')
            plt.savefig(file, dpi=300, bbox_inches='tight', facecolor='#0d1117')
            plt.show()

        plot("FIGURA 13 - Voz Original", self.audio, "#58a6ff", "fig13.png")
        plot("FIGURA 14 - Impulso", audio2, "#ffa657", "fig14.png")
        plot("FIGURA 15 - Reverb", conv, "#f85149", "fig15.png")
        messagebox.showinfo("FIGURA 16", "Se reproducen los 3 audios")
        threading.Thread(target=self.play_audio, args=(self.audio_path,)).start()
        time.sleep(2)
        threading.Thread(target=self.play_audio, args=(path,)).start()
        time.sleep(2)
        threading.Thread(target=self.play_audio, args=("reverb.wav",)).start()

    def apply_sampling_audio(self):
        if self.audio is None: return
        down = self.audio[::2]
        wavfile.write("grave.wav", 22050, down.astype(np.int16))
        plt.figure(figsize=(14,7), facecolor='#0d1117')
        plt.subplot(121); self.plot_spec(self.audio,44100,"Original","#58a6ff")
        plt.subplot(122); self.plot_spec(down,22050,"Downsampling","#ff7b72")
        plt.suptitle("FIGURA 17", color='white', fontsize=16)
        plt.savefig("fig17.png", dpi=300)
        plt.show()
        self.play_audio(self.audio_path)
        time.sleep(1)
        self.play_audio("grave.wav")

    def plot_spec(self, data, rate, title, color):
        f = np.fft.fftfreq(len(data), 1/rate)
        s = np.abs(np.fft.fft(data))
        plt.plot(f[:len(f)//2], s[:len(s)//2], color, linewidth=2)
        plt.title(title, color='white')
        plt.xlim(0,8000)
        plt.gca().set_facecolor('#161b22')

    def apply_laplace_to_video(self):
        if not self.video_frames: 
            messagebox.showerror("Error", "Carga un video")
            return
        frame = self.video_frames[0]
        filtered = signal.convolve2d(frame, kernels['Laplace'], mode='same')
        plt.figure(figsize=(14,7), facecolor='#0d1117')
        plt.subplot(121); plt.imshow(frame, cmap='gray'); plt.title("Original", color='white'); plt.axis('off')
        plt.subplot(122); plt.imshow(np.abs(filtered), cmap='gray'); plt.title("FIGURA 20 - Laplace", color='white'); plt.axis('off')
        plt.savefig("fig20.png", dpi=300)
        plt.show()

    def apply_sampling_video(self):
        if not self.video_frames: 
            messagebox.showerror("Error", "Carga un video")
            return
        frame = self.video_frames[0]
        down = frame[::4, ::4]
        plt.figure(figsize=(14,7), facecolor='#0d1117')
        plt.subplot(121); plt.imshow(frame, cmap='gray'); plt.title("Original", color='white'); plt.axis('off')
        plt.subplot(122); plt.imshow(down, cmap='gray'); plt.title("FIGURA 21 - Muestreado", color='white'); plt.axis('off')
        plt.savefig("fig21.png", dpi=300)
        plt.show()

    def play_audio(self, path):
        if path and os.path.exists(path):
            play(AudioSegment.from_file(path))

    def play_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "No hay video")
            return
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow('Video', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessor(root)
    root.mainloop()