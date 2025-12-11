# Proyecto Integrador: Aplicación de Procesamiento de Señales Multimedia
# Requiere: pip install opencv-python numpy scipy matplotlib pyaudio pydub tkinter

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

# Definir kernels para filtros
kernels = {
    'Roberts': np.array([[1, 0], [0, -1]]),
    'Prewitt Horizontal': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    'Prewitt Vertical': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    'Sobel Horizontal': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    'Sobel Vertical': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    'Sobel Ambos': (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])),
    'Laplace': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    'LoG': np.array([[0, 0, -1, 0, 0],
                     [0, -1, -2, -1, 0],
                     [-1, -2, 16, -2, -1],
                     [0, -1, -2, -1, 0],
                     [0, 0, -1, 0, 0]]) / 16
}

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Señales Multimedia")
        self.root.geometry("800x600")
        self.root.configure(bg="#0d1117")

        self.photo = None
        self.audio_data = None
        self.audio_rate = None
        self.audio_path = None
        self.video_frames = []
        self.video_path = None

        self.show_main_menu()

    def show_main_menu(self):
        self.clear_menu()
        tk.Label(self.root, text="Seleccione el tipo de señal:", bg="#0d1117", fg="#ffffff", font=("Arial", 14)).pack(pady=20)
        tk.Button(self.root, text="Fotografía", command=self.show_photo_menu, width=20).pack(pady=10)
        tk.Button(self.root, text="Clip de Audio", command=self.show_audio_menu, width=20).pack(pady=10)
        tk.Button(self.root, text="Clip de Video", command=self.show_video_menu, width=20).pack(pady=10)

    def show_photo_menu(self):
        self.clear_menu()
        tk.Label(self.root, text="Menú de Fotografía", bg="#0d1117", fg="#ffffff", font=("Arial", 14)).pack(pady=20)
        tk.Button(self.root, text="Capturar Fotografía", command=self.capture_photo, width=20).pack(pady=5)
        tk.Button(self.root, text="Cargar Fotografía", command=self.load_photo, width=20).pack(pady=5)
        tk.Button(self.root, text="Mostrar Original", command=lambda: self.show_image(self.photo, "Original"), width=20).pack(pady=5)
        for name in kernels:
            tk.Button(self.root, text=f"Convolución {name}", command=lambda n=name: self.apply_convolution(self.photo, n), width=20).pack(pady=5)
        tk.Button(self.root, text="FFT e IFFT", command=self.apply_fft_ifft_photo, width=20).pack(pady=5)
        tk.Button(self.root, text="Muestreo", command=self.apply_sampling_photo, width=20).pack(pady=5)
        tk.Button(self.root, text="Volver", command=self.show_main_menu, width=20).pack(pady=5)

    def show_audio_menu(self):
        self.clear_menu()
        tk.Label(self.root, text="Menú de Audio", bg="#0d1117", fg="#ffffff", font=("Arial", 14)).pack(pady=20)
        tk.Button(self.root, text="Capturar Audio", command=self.capture_audio, width=20).pack(pady=5)
        tk.Button(self.root, text="Cargar Audio", command=self.load_audio, width=20).pack(pady=5)
        tk.Button(self.root, text="Mostrar Espectro Original", command=self.show_spectrum, width=20).pack(pady=5)
        tk.Button(self.root, text="Convolución de Sonidos", command=self.convolve_audio, width=20).pack(pady=5)
        tk.Button(self.root, text="FFT e IFFT", command=self.apply_fft_ifft_audio, width=20).pack(pady=5)
        tk.Button(self.root, text="Muestreo", command=self.apply_sampling_audio, width=20).pack(pady=5)
        tk.Button(self.root, text="Volver", command=self.show_main_menu, width=20).pack(pady=5)

    def show_video_menu(self):
        self.clear_menu()
        tk.Label(self.root, text="Menú de Video", bg="#0d1117", fg="#ffffff", font=("Arial", 14)).pack(pady=20)
        tk.Button(self.root, text="Capturar Video", command=self.capture_video, width=20).pack(pady=5)
        tk.Button(self.root, text="Cargar Video", command=self.load_video, width=20).pack(pady=5)
        tk.Button(self.root, text="Mostrar Frame Original", command=lambda: self.show_image(self.video_frames[0] if self.video_frames else None, "Frame Original"), width=20).pack(pady=5)
        for name in kernels:
            tk.Button(self.root, text=f"Convolución {name} a Frames", command=lambda n=name: self.apply_convolution_video(n), width=20).pack(pady=5)
        tk.Button(self.root, text="FFT e IFFT a Frame", command=self.apply_fft_ifft_video, width=20).pack(pady=5)
        tk.Button(self.root, text="Muestreo a Frames", command=self.apply_sampling_video, width=20).pack(pady=5)
        tk.Button(self.root, text="Volver", command=self.show_main_menu, width=20).pack(pady=5)

    def clear_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def capture_photo(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            self.photo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cap.release()
            messagebox.showinfo("Éxito", "Fotografía capturada")
        else:
            messagebox.showerror("Error", "No se pudo capturar")

    def load_photo(self):
        path = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.photo = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            messagebox.showinfo("Éxito", "Fotografía cargada")

    def capture_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        frames = []
        for _ in range(0, int(44100 / 1024 * 5)):
            data = stream.read(1024)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

        self.audio_path = "audio_capturado.wav"
        wf = wave.open(self.audio_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.audio_rate, self.audio_data = wavfile.read(self.audio_path)
        messagebox.showinfo("Éxito", "Audio capturado")

    def load_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
        if path:
            if path.endswith('.mp3'):
                audio = AudioSegment.from_mp3(path)
                path = "audio_cargado.wav"
                audio.export(path, format="wav")
            self.audio_path = path
            self.audio_rate, self.audio_data = wavfile.read(path)
            messagebox.showinfo("Éxito", "Audio cargado")

    def capture_video(self):
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_path = 'video_capturado.avi'
        out = cv2.VideoWriter(self.video_path, fourcc, 20.0, (640, 480))

        self.video_frames = []
        for _ in range(100):  # 5 segundos approx
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                self.video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                break

        cap.release()
        out.release()
        if self.video_frames:
            messagebox.showinfo("Éxito", "Video capturado")
        else:
            messagebox.showerror("Error", "No se pudo capturar")

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.avi *.mp4 *.mov")])
        if path:
            cap = cv2.VideoCapture(path)
            self.video_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cap.release()
            self.video_path = path
            if self.video_frames:
                messagebox.showinfo("Éxito", "Video cargado")
            else:
                messagebox.showerror("Error", "No se pudo cargar")

    def show_image(self, img, title):
        if img is None:
            messagebox.showerror("Error", "No hay señal cargada")
            return
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.show()

    def apply_convolution(self, img, name):
        if img is None:
            messagebox.showerror("Error", "No hay imagen")
            return
        kernel = kernels[name]
        if name == 'Sobel Ambos':
            filtered_x = signal.convolve2d(img, kernel[0], mode='same')
            filtered_y = signal.convolve2d(img, kernel[1], mode='same')
            filtered = np.sqrt(filtered_x**2 + filtered_y**2)
        else:
            filtered = signal.convolve2d(img, kernel, mode='same')
        self.show_image(img, "Original")
        self.show_image(np.abs(filtered), f"Filtrado con {name}")

    def apply_fft_ifft_photo(self):
        if self.photo is None:
            messagebox.showerror("Error", "No hay foto")
            return
        fft = np.fft.fft2(self.photo)
        ifft = np.fft.ifft2(fft).real
        self.show_image(self.photo, "Original")
        plt.imshow(np.log(np.abs(np.fft.fftshift(fft)) + 1), cmap='gray')
        plt.title("FFT")
        plt.show()
        self.show_image(ifft, "IFFT")

    def apply_sampling_photo(self):
        if self.photo is None:
            messagebox.showerror("Error", "No hay foto")
            return
        downsampled = self.photo[::2, ::2]
        self.show_image(self.photo, "Original")
        self.show_image(downsampled, "Muestreada")

    def show_spectrum(self):
        if self.audio_data is None:
            messagebox.showerror("Error", "No hay audio")
            return
        freqs = np.fft.fftfreq(len(self.audio_data), 1/self.audio_rate)
        spectrum = np.abs(np.fft.fft(self.audio_data))
        plt.plot(freqs[:len(freqs)//2], spectrum[:len(spectrum)//2])
        plt.title("Espectro Original")
        plt.show()

    def convolve_audio(self):
        if self.audio_data is None:
            messagebox.showerror("Error", "No hay audio")
            return
        path = filedialog.askopenfilename(title="Seleccione otro audio", filetypes=[("Audio", "*.wav *.mp3")])
        if path:
            if path.endswith('.mp3'):
                audio = AudioSegment.from_mp3(path)
                path = "audio2.wav"
                audio.export(path, format="wav")
            rate2, audio2 = wavfile.read(path)
            if rate2 != self.audio_rate:
                messagebox.showerror("Error", "Tasas de muestreo diferentes")
                return
            convolved = signal.convolve(self.audio_data, audio2)
            convolved_path = "convolved.wav"
            wavfile.write(convolved_path, self.audio_rate, convolved.astype(np.int16))

            # Espectros
            self.show_spectrum()  # Figura 13
            freqs2 = np.fft.fftfreq(len(audio2), 1/rate2)
            spectrum2 = np.abs(np.fft.fft(audio2))
            plt.plot(freqs2[:len(freqs2)//2], spectrum2[:len(spectrum2)//2])
            plt.title("Espectro Audio 2 (Figura 14)")
            plt.show()

            freqs_conv = np.fft.fftfreq(len(convolved), 1/self.audio_rate)
            spectrum_conv = np.abs(np.fft.fft(convolved))
            plt.plot(freqs_conv[:len(freqs_conv)//2], spectrum_conv[:len(spectrum_conv)//2])
            plt.title("Espectro Convolucionado (Figura 15)")
            plt.show()

            # Reproducción (Figura 16)
            threading.Thread(target=self.play_audio, args=(self.audio_path,)).start()
            time.sleep(1)
            threading.Thread(target=self.play_audio, args=(path,)).start()
            time.sleep(1)
            threading.Thread(target=self.play_audio, args=(convolved_path,)).start()

    def apply_fft_ifft_audio(self):
        if self.audio_data is None:
            messagebox.showerror("Error", "No hay audio")
            return
        fft = np.fft.fft(self.audio_data)
        ifft = np.fft.ifft(fft).real
        ifft_path = "ifft_audio.wav"
        wavfile.write(ifft_path, self.audio_rate, ifft.astype(np.int16))
        self.play_audio(ifft_path)

    def apply_sampling_audio(self):
        if self.audio_data is None:
            messagebox.showerror("Error", "No hay audio")
            return
        downsampled = self.audio_data[::2]
        downsampled_path = "downsampled_audio.wav"
        wavfile.write(downsampled_path, self.audio_rate // 2, downsampled.astype(np.int16))
        # Figura 17
        plt.figure(figsize=(12, 6))
        freqs = np.fft.fftfreq(len(self.audio_data), 1/self.audio_rate)
        spectrum = np.abs(np.fft.fft(self.audio_data))
        plt.subplot(1, 2, 1)
        plt.plot(freqs[:len(freqs)//2], spectrum[:len(spectrum)//2])
        plt.title("Audio Original")

        freqs_down = np.fft.fftfreq(len(downsampled), 1/(self.audio_rate // 2))
        spectrum_down = np.abs(np.fft.fft(downsampled))
        plt.subplot(1, 2, 2)
        plt.plot(freqs_down[:len(freqs_down)//2], spectrum_down[:len(spectrum_down)//2])
        plt.title("Audio Muestreado (más grave)")
        plt.show()
        self.play_audio(downsampled_path)

    def apply_convolution_video(self, name):
        if not self.video_frames:
            messagebox.showerror("Error", "No hay video")
            return
        filtered_frames = []
        for frame in self.video_frames:
            self.apply_convolution(frame, name)  # Muestra originales y filtradas
        messagebox.showinfo("Info", "Convolución aplicada a frames")

    def apply_fft_ifft_video(self):
        if not self.video_frames:
            messagebox.showerror("Error", "No hay video")
            return
        self.apply_fft_ifft_photo()  # Aplica a primer frame

    def apply_sampling_video(self):
        if not self.video_frames:
            messagebox.showerror("Error", "No hay video")
            return
        self.apply_sampling_photo()  # Aplica a primer frame

    def play_audio(self, path):
        sound = AudioSegment.from_file(path)
        play(sound)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()