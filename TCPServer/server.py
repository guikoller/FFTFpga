# server.py (versão atualizada)
import socket
import threading
import tkinter as tk
from tkinter import ttk, font, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import wave
import struct
import time

# --- Constantes ---
HOST = '127.0.0.1'
PORT = 65432
CHUNK = 1024 * 2
RATE = 44100  # Taxa de amostragem padrão

class ServerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Servidor de Análise FFT v2.0")
        self.master.geometry("850x750")
        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)

        # --- Variáveis de estado ---
        self.server_socket = None
        self.client_conn = None
        self.server_thread = None
        self.streaming_thread = None
        
        self.is_server_running = False
        self.is_streaming = False
        self.source_mode = tk.StringVar(value="Onda Senoidal")
        
        # Variáveis para geração de onda
        self.sine_frequency = tk.DoubleVar(value=440.0) # Frequência em Hz (A4)
        self.sine_amplitude = tk.DoubleVar(value=15000.0) # Amplitude
        self.sine_phase = 0

        # Variáveis para arquivo de áudio
        self.audio_data = None
        self.audio_pos = 0

        self._setup_ui()
        self._update_status("Servidor Desligado", "red")
        self._toggle_source_controls() # Configura estado inicial dos controles

    def _setup_ui(self):
        # --- Frame de Controle Principal ---
        main_control_frame = ttk.Frame(self.master, padding="10")
        main_control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # --- Coluna 1: Servidor e Status ---
        server_frame = ttk.LabelFrame(main_control_frame, text="Controle do Servidor", padding="10")
        server_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.btn_toggle_server = ttk.Button(server_frame, text="Ligar Servidor", command=self.toggle_server)
        self.btn_toggle_server.pack(pady=5, fill=tk.X)

        self.btn_toggle_stream = ttk.Button(server_frame, text="Iniciar Envio", state=tk.DISABLED, command=self.toggle_streaming)
        self.btn_toggle_stream.pack(pady=5, fill=tk.X)
        
        status_frame = ttk.Frame(server_frame)
        status_frame.pack(pady=10, fill=tk.X)
        ttk.Label(status_frame, text="Status:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.status_value = ttk.Label(status_frame, text="", font=("Helvetica", 10))
        self.status_value.pack(side=tk.LEFT)

        # --- Coluna 2: Fonte do Sinal ---
        source_frame = ttk.LabelFrame(main_control_frame, text="Fonte do Sinal", padding="10")
        source_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Radiobutton(source_frame, text="Onda Senoidal", variable=self.source_mode, value="Onda Senoidal", command=self._toggle_source_controls).pack(anchor=tk.W)
        ttk.Radiobutton(source_frame, text="Arquivo de Áudio (.wav)", variable=self.source_mode, value="Arquivo de Áudio", command=self._toggle_source_controls).pack(anchor=tk.W)

        # --- Coluna 3: Controles da Fonte ---
        controls_frame = ttk.LabelFrame(main_control_frame, text="Configurações da Fonte", padding="10")
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Controles da Senoide
        self.sine_controls_frame = ttk.Frame(controls_frame)
        ttk.Label(self.sine_controls_frame, text="Frequência (Hz):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(self.sine_controls_frame, textvariable=self.sine_frequency, width=10).grid(row=0, column=1, padx=5)
        ttk.Label(self.sine_controls_frame, text="Amplitude:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Entry(self.sine_controls_frame, textvariable=self.sine_amplitude, width=10).grid(row=1, column=1, padx=5)
        
        # Controles do Arquivo
        self.file_controls_frame = ttk.Frame(controls_frame)
        self.btn_load_file = ttk.Button(self.file_controls_frame, text="Carregar Arquivo .wav", command=self.load_audio_file)
        self.btn_load_file.pack()
        self.file_label = ttk.Label(self.file_controls_frame, text="Nenhum arquivo carregado.", wraplength=200)
        self.file_label.pack(pady=5)
        
        # --- Gráficos ---
        plot_frame = ttk.Frame(self.master, padding="10")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig, (self.ax_time, self.ax_fft) = plt.subplots(2, 1, figsize=(8, 6))
        
        # Gráfico do Sinal Original
        self.ax_time.set_title("Sinal Original Enviado")
        self.ax_time.set_xlabel("Amostras")
        self.ax_time.set_ylabel("Amplitude")
        self.line_time, = self.ax_time.plot([], [], lw=1)
        
        # Gráfico da FFT
        self.ax_fft.set_title("FFT Recebida do Cliente")
        self.ax_fft.set_xlabel("Frequência (Hz)")
        self.ax_fft.set_ylabel("Magnitude")
        self.freq_bins = np.fft.fftfreq(CHUNK, 1/RATE)
        self.line_fft, = self.ax_fft.plot([], [], lw=1, color='orange')
        
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _toggle_source_controls(self):
        """Alterna a visibilidade dos controles com base na fonte selecionada."""
        if self.source_mode.get() == "Onda Senoidal":
            self.sine_controls_frame.pack(fill=tk.X, expand=True)
            self.file_controls_frame.pack_forget()
        else:
            self.sine_controls_frame.pack_forget()
            self.file_controls_frame.pack(fill=tk.X, expand=True)

    def _update_status(self, text, color):
        self.status_value.config(text=text, foreground=color)

    def toggle_server(self):
        if self.is_server_running:
            self.stop_server()
        else:
            self.start_server()

    def start_server(self):
        self.is_server_running = True
        self.btn_toggle_server.config(text="Desligar Servidor")
        self._update_status("Ouvindo...", "orange")
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()

    def stop_server(self):
        if self.is_streaming:
            self.stop_streaming()
        
        self.is_server_running = False
        self.btn_toggle_server.config(text="Ligar Servidor")
        self._update_status("Servidor Desligado", "red")
        self.btn_toggle_stream.config(state=tk.DISABLED)

        if self.client_conn:
            self.client_conn.close()
        if self.server_socket:
            try:
                # Desbloqueia o accept()
                dummy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                dummy_socket.connect((HOST, PORT))
                dummy_socket.close()
            except Exception:
                pass
            self.server_socket.close()
        self.client_conn = None

    def _server_loop(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((HOST, PORT))
            self.server_socket.listen()
            
            while self.is_server_running:
                try:
                    conn, addr = self.server_socket.accept()
                    self.client_conn = conn
                    self.master.after(0, self._update_status, f"Conectado com {addr[0]}", "green")
                    self.master.after(0, self.btn_toggle_stream.config, {"state": tk.NORMAL})
                    
                    # Mantém a thread viva esperando o cliente desconectar
                    while self.is_server_running:
                        # Uma forma simples de verificar se o cliente ainda está conectado
                        test_data = self.client_conn.recv(1, socket.MSG_PEEK)
                        if not test_data:
                            break
                        time.sleep(0.5)

                except OSError:
                    break # Socket foi fechado
                finally:
                    if self.client_conn:
                        self.client_conn.close()
                        self.client_conn = None
                    if self.is_server_running:
                         self.master.after(0, self._update_status, "Ouvindo...", "orange")
                         self.master.after(0, self.btn_toggle_stream.config, {"state": tk.DISABLED})

        finally:
            self.master.after(0, self.stop_server)

    def load_audio_file(self):
        filepath = filedialog.askopenfilename(
            title="Selecionar arquivo de áudio",
            filetypes=(("Arquivos WAV", "*.wav"), ("Todos os arquivos", "*.*"))
        )
        if not filepath:
            return

        try:
            with wave.open(filepath, 'rb') as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    messagebox.showerror("Erro de Formato", "O arquivo deve ser mono, 16-bit.")
                    return
                
                global RATE
                RATE = wf.getframerate()
                self.audio_data = wf.readframes(wf.getnframes())
                self.audio_pos = 0
                self.file_label.config(text=filepath.split('/')[-1])
                print(f"Arquivo carregado: {len(self.audio_data)} bytes, Taxa: {RATE} Hz")
        except Exception as e:
            messagebox.showerror("Erro ao Ler Arquivo", f"Não foi possível ler o arquivo:\n{e}")

    def get_next_chunk(self):
        """Gera ou lê o próximo bloco de dados de áudio."""
        if self.source_mode.get() == "Onda Senoidal":
            t = (self.sine_phase + np.arange(CHUNK)) / RATE
            freq = self.sine_frequency.get()
            amp = self.sine_amplitude.get()
            
            wave_data = amp * np.sin(2 * np.pi * freq * t)
            self.sine_phase += CHUNK # Atualiza a fase para a próxima chamada
            return wave_data.astype(np.int16)

        elif self.source_mode.get() == "Arquivo de Áudio":
            if self.audio_data is None:
                return None # Nenhum arquivo carregado

            if self.audio_pos + CHUNK * 2 > len(self.audio_data):
                # Fim do arquivo, reinicia
                self.audio_pos = 0

            end_pos = self.audio_pos + CHUNK * 2
            chunk_bytes = self.audio_data[self.audio_pos:end_pos]
            self.audio_pos = end_pos
            
            # Converte bytes para numpy array
            return np.frombuffer(chunk_bytes, dtype=np.int16)
        return None

    def toggle_streaming(self):
        if self.is_streaming:
            self.stop_streaming()
        else:
            self.start_streaming()

    def start_streaming(self):
        if self.source_mode.get() == "Arquivo de Áudio" and self.audio_data is None:
            messagebox.showwarning("Aviso", "Carregue um arquivo de áudio antes de iniciar o envio.")
            return

        self.is_streaming = True
        self.btn_toggle_stream.config(text="Parar Envio")
        self.streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.streaming_thread.start()

    def stop_streaming(self):
        self.is_streaming = False
        self.btn_toggle_stream.config(text="Iniciar Envio")

    def _streaming_loop(self):
        while self.is_streaming and self.client_conn:
            try:
                # 1. Obter o próximo bloco de dados
                signal_chunk = self.get_next_chunk()
                if signal_chunk is None or len(signal_chunk) < CHUNK:
                    time.sleep(0.1) # Espera se não houver dados
                    continue
                
                # 2. Enviar para o cliente
                self.client_conn.sendall(signal_chunk.tobytes())

                # 3. Receber a FFT de volta
                fft_bytes = self.client_conn.recv(CHUNK * 4) # 4 bytes por float
                if not fft_bytes:
                    break # Cliente desconectou
                
                fft_data = np.frombuffer(fft_bytes, dtype=np.float32)

                # 4. Agendar a atualização dos gráficos na thread principal
                self.master.after(0, self.update_plots, signal_chunk, fft_data)
                
                # Controla a taxa de envio para corresponder à taxa de amostragem
                time.sleep(CHUNK / RATE * 0.9)

            except (ConnectionResetError, BrokenPipeError):
                print("Conexão com o cliente perdida durante o streaming.")
                self.master.after(0, self.stop_streaming)
                break
            except Exception as e:
                print(f"Erro no loop de streaming: {e}")
                self.master.after(0, self.stop_streaming)
                break

    def update_plots(self, signal_data, fft_data):
        # Atualiza gráfico do sinal original
        self.ax_time.set_ylim(np.min(signal_data) * 1.1, np.max(signal_data) * 1.1)
        self.line_time.set_data(np.arange(len(signal_data)), signal_data)
        self.ax_time.set_xlim(0, len(signal_data))

        # Atualiza gráfico da FFT
        # Usamos apenas a primeira metade dos dados, que é a parte útil
        valid_fft_data = fft_data[:CHUNK//2]
        valid_freq_bins = self.freq_bins[:CHUNK//2]
        
        self.line_fft.set_data(valid_freq_bins, valid_fft_data)
        self.ax_fft.set_xlim(0, RATE / 2) # Limita a visualização até a frequência de Nyquist
        max_fft = np.max(valid_fft_data)
        if max_fft > self.ax_fft.get_ylim()[1] or max_fft < self.ax_fft.get_ylim()[1] / 2:
            self.ax_fft.set_ylim(0, max_fft * 1.2 if max_fft > 0 else 1)

        self.canvas.draw()

    def _on_closing(self):
        self.stop_server()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ServerApp(root)
    root.mainloop()