# gui_app.py
import tkinter as tk
from tkinter import ttk, font, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import queue

from server import Server

class GUIApplication:
    """Manages the entire Graphical User Interface (GUI) for the server control panel."""
    def __init__(self, master):
        self.master = master
        self.master.title("FPGA FFT Test Server Control Panel")
        self.master.geometry("1050x750")
        
        self.data_queue = queue.Queue()
        
        # Pass the new progress_callback to the Server instance
        self.server = Server(
            data_queue=self.data_queue,
            status_callback=self.update_status,
            stream_stopped_callback=self.on_stream_stopped,
            progress_callback=self.update_progress 
        )
        
        self._setup_ui()
        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.master.after(100, self.periodic_data_check)
        self.update_status("Offline", "red")

    def _setup_ui(self):
        # --- Main Control Frame ---
        main_control_frame = ttk.Frame(self.master, padding="10")
        main_control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # --- Column 1 & 2: Server Control ---
        server_frame = ttk.LabelFrame(main_control_frame, text="Server Control", padding="10")
        server_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), anchor='n')
        
        config_frame = ttk.Frame(server_frame, padding=(0,0,10,0))
        config_frame.pack(side=tk.LEFT, fill=tk.Y)
        action_frame = ttk.Frame(server_frame)
        action_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(config_frame, text="Server IP Address:").pack(anchor='w', pady=(0, 2))
        self.ip_address = tk.StringVar(value="127.0.0.1")
        self.ip_entry = ttk.Entry(config_frame, textvariable=self.ip_address)
        self.ip_entry.pack(fill='x', pady=(0, 10))
        ttk.Label(config_frame, text="Server Port:").pack(anchor='w', pady=(0, 2))
        self.port = tk.StringVar(value="65432")
        self.port_entry = ttk.Entry(config_frame, textvariable=self.port)
        self.port_entry.pack(fill='x', pady=(0, 15))
        self.btn_toggle_server = ttk.Button(action_frame, text="Start Server", command=self.toggle_server)
        self.btn_toggle_server.pack(pady=5, fill=tk.X, ipadx=10)
        self.btn_toggle_stream = ttk.Button(action_frame, text="Start Streaming", state=tk.DISABLED, command=self.toggle_streaming)
        self.btn_toggle_stream.pack(pady=5, fill=tk.X, ipadx=10)
        status_frame = ttk.Frame(action_frame)
        status_frame.pack(pady=10, fill=tk.X, side='bottom', expand=True)
        ttk.Label(status_frame, text="Status:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.status_value = ttk.Label(status_frame, text="", font=("Helvetica", 10))
        self.status_value.pack(side=tk.LEFT)

        # --- Column 3: Signal Source Selection ---
        source_frame = ttk.LabelFrame(main_control_frame, text="Signal Source", padding="10")
        source_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, anchor='n')
        self.source_mode = tk.StringVar(value="Audio File")
        ttk.Radiobutton(source_frame, text="Sine Wave", variable=self.source_mode, value="Sine Wave", command=self._toggle_source_controls).pack(anchor=tk.W)
        ttk.Radiobutton(source_frame, text="Audio File (.wav)", variable=self.source_mode, value="Audio File", command=self._toggle_source_controls).pack(anchor=tk.W)

        # --- Column 4: Source Settings ---
        controls_frame = ttk.LabelFrame(main_control_frame, text="Source Settings", padding="10")
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, anchor='n')

        self.sine_controls_frame = ttk.Frame(controls_frame)
        self.sine_frequency = tk.DoubleVar(value=440.0)
        self.sine_amplitude = tk.DoubleVar(value=15000.0)
        self.sine_duration = tk.DoubleVar(value=0) # Duration for sine wave
        ttk.Label(self.sine_controls_frame, text="Frequency (Hz):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(self.sine_controls_frame, textvariable=self.sine_frequency, width=12).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(self.sine_controls_frame, text="Amplitude:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Entry(self.sine_controls_frame, textvariable=self.sine_amplitude, width=12).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(self.sine_controls_frame, text="Duration (s):").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(self.sine_controls_frame, textvariable=self.sine_duration, width=12).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(self.sine_controls_frame, text="(0 = infinite)").grid(row=3, column=1, sticky='w', padx=5)
        
        self.file_controls_frame = ttk.Frame(controls_frame)
        self.btn_load_file = ttk.Button(self.file_controls_frame, text="Load .wav File", command=self.load_file)
        self.btn_load_file.pack(fill='x')
        self.file_label = ttk.Label(self.file_controls_frame, text="No file loaded.", wraplength=200)
        self.file_label.pack(pady=5)
        
        # Add Progress Bar and Label
        self.progress_var = tk.DoubleVar()
        self.progress_label_var = tk.StringVar(value="0%")
        self.progress_bar = ttk.Progressbar(self.file_controls_frame, orient="horizontal", length=200, mode="determinate", variable=self.progress_var)
        self.progress_bar.pack(pady=(10,0), fill='x')
        self.progress_label = ttk.Label(self.file_controls_frame, textvariable=self.progress_label_var, anchor='center')
        self.progress_label.pack(fill='x')
        
        self._toggle_source_controls()
        
        # --- Plotting Area ---
        plot_frame = ttk.Frame(self.master, padding="10")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig, (self.ax_time, self.ax_fft) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
        self.ax_time.set_title("Original Signal Sent to FPGA")
        self.ax_fft.set_title("FFT Result Received from FPGA")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_progress(self, percentage):
        """Callback function to update the progress bar and label."""
        self.progress_var.set(percentage)
        self.progress_label_var.set(f"{percentage:.0f}%")

    def _toggle_source_controls(self):
        if self.source_mode.get() == "Sine Wave":
            self.sine_controls_frame.pack(pady=5)
            self.file_controls_frame.pack_forget()
        else: # Audio File
            self.sine_controls_frame.pack_forget()
            self.file_controls_frame.pack(pady=5)

    def on_stream_stopped(self):
        self.btn_toggle_stream.config(text="Start Streaming")

    def update_status(self, text, color):
        self.status_value.config(text=text, foreground=color)
        is_connected = "Connected" in text
        self.btn_toggle_stream.config(state=tk.NORMAL if is_connected else tk.DISABLED)
        if not self.server.is_streaming:
            self.on_stream_stopped()

    def toggle_server(self):
        if self.server.is_server_running:
            self.server.stop()
            self.btn_toggle_server.config(text="Start Server")
            self.ip_entry.config(state=tk.NORMAL)
            self.port_entry.config(state=tk.NORMAL)
        else:
            host = self.ip_address.get()
            port = int(self.port.get())
            self.server.start(host, port)
            self.btn_toggle_server.config(text="Stop Server")
            self.ip_entry.config(state=tk.DISABLED)
            self.port_entry.config(state=tk.DISABLED)
            
    def toggle_streaming(self):
        if self.server.is_streaming:
            self.server.stop_streaming()
        else:
            config = {
                'mode': self.source_mode.get(),
                'freq': self.sine_frequency.get(),
                'amp': self.sine_amplitude.get(),
                'duration': self.sine_duration.get()
            }
            if config['mode'] == "Audio File" and self.server.audio_data is None:
                messagebox.showwarning("Warning", "Please load an audio file before starting.")
                return
            
            self.server.start_streaming(config)
            self.btn_toggle_stream.config(text="Stop Streaming")

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=(("WAV Files", "*.wav"),))
        if not filepath: return
        
        metadata, error = self.server.load_audio_file(filepath)
        if error:
            messagebox.showerror("File Error", error)
        else:
            self.file_label.config(text=filepath.split('/')[-1])
            print(f"File loaded successfully. Rate: {metadata['rate']} Hz")
    
    def periodic_data_check(self):
        try:
            if not self.data_queue.empty():
                signal_data, fft_data, rate = self.data_queue.get_nowait()
                self.update_plots(signal_data, fft_data, rate)
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.periodic_data_check)

    def update_plots(self, signal_data, fft_data, rate):
        chunk_size = len(signal_data)
        self.ax_time.clear()
        self.ax_time.set_title("Original Signal Sent to FPGA")
        self.ax_time.set_xlabel("Samples")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.plot(signal_data, lw=1, color='royalblue')
        self.ax_time.grid(True, linestyle='--', alpha=0.6)
        self.ax_fft.clear()
        self.ax_fft.set_title("FFT Result Received from FPGA")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Magnitude")
        freq_bins = np.fft.fftfreq(chunk_size, 1/rate)
        self.ax_fft.plot(freq_bins[:chunk_size//2], fft_data[:chunk_size//2], lw=1, color='darkorange')
        self.ax_fft.set_xlim(0, rate / 2)
        self.ax_fft.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()
        
    def _on_closing(self):
        self.server.stop()
        self.master.destroy()