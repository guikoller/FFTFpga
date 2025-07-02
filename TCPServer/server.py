# server.py
import socket
import threading
import numpy as np
import wave
import time

class Server:
    """Manages the server's networking logic and data generation, decoupled from the GUI."""

    def __init__(self, data_queue, status_callback, stream_stopped_callback, progress_callback):
        self.data_queue = data_queue
        self.status_callback = status_callback
        self.stream_stopped_callback = stream_stopped_callback
        self.progress_callback = progress_callback # New callback for progress updates
        
        self.chunk_size = 1024 * 2
        self.rate = 44100

        self.server_socket = None
        self.client_conn = None
        self.server_thread = None
        self.streaming_thread = None
        
        self.is_server_running = False
        self.is_streaming = False

        self.audio_data = None
        self.audio_pos = 0
        self.sine_phase = 0
        self.streaming_config = {}

    def start(self, host, port):
        if self.is_server_running: return
        self.is_server_running = True
        self.server_thread = threading.Thread(target=self._server_loop, args=(host, port), daemon=True)
        self.server_thread.start()

    def stop(self):
        if not self.is_server_running: return
        self.stop_streaming()
        self.is_server_running = False
        
        if self.client_conn: self.client_conn.close()
        
        if self.server_socket:
            try:
                host, port = self.server_socket.getsockname()
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            except Exception:
                pass
            self.server_socket.close()
        
        self.status_callback("Offline", "red")

    def _server_loop(self, host, port):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((host, port))
            self.server_socket.listen()
            self.status_callback(f"Listening on {host}:{port}", "orange")
            
            while self.is_server_running:
                conn, addr = self.server_socket.accept()
                if not self.is_server_running: break
                
                self.client_conn = conn
                self.status_callback(f"Connected to {addr[0]}", "green")
                while self.is_server_running:
                    peek_data = self.client_conn.recv(1, socket.MSG_PEEK)
                    if not peek_data: break
                    time.sleep(0.5)
                self.client_conn = None
                if self.is_server_running:
                    self.status_callback(f"Listening on {host}:{port}", "orange")

        except OSError as e:
            if e.errno == 98:
                self.status_callback(f"Error: Port {port} in use", "red")
            else:
                 self.status_callback("Server Stopped", "red")
        except Exception as e:
            self.status_callback(f"Error: {e}", "red")
        finally:
            self.is_server_running = False
            self.status_callback("Offline", "red")

    def start_streaming(self, config):
        if self.is_streaming or not self.client_conn: return
        self.is_streaming = True
        self.streaming_config = config
        
        # Reset progress bar when starting to stream a file
        if self.streaming_config['mode'] == "Audio File":
            self.progress_callback(0)

        self.streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.streaming_thread.start()

    def stop_streaming(self):
        self.is_streaming = False
        
    def load_audio_file(self, filepath):
        try:
            with wave.open(filepath, 'rb') as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    return None, "Error: File must be mono, 16-bit WAV."
                self.rate = wf.getframerate()
                self.audio_data = wf.readframes(wf.getnframes())
                self.audio_pos = 0
                self.progress_callback(0) # Reset progress bar on new file load
                return {'rate': self.rate}, None
        except Exception as e:
            return None, f"Could not read file:\n{e}"

    def _get_next_chunk(self):
        mode = self.streaming_config.get('mode')
        if mode == "Sine Wave":
            freq = self.streaming_config.get('freq', 440.0)
            amp = self.streaming_config.get('amp', 15000.0)
            t = (self.sine_phase + np.arange(self.chunk_size)) / self.rate
            wave_data = amp * np.sin(2 * np.pi * freq * t)
            self.sine_phase += self.chunk_size
            return wave_data.astype(np.int16)
        elif mode == "Audio File":
            if self.audio_data is None: return None
            # Stop streaming if we've reached the end of the file
            if self.audio_pos >= len(self.audio_data):
                self.progress_callback(100)
                return None

            end_pos = self.audio_pos + self.chunk_size * 2
            chunk_bytes = self.audio_data[self.audio_pos:end_pos]
            self.audio_pos = end_pos
            return np.frombuffer(chunk_bytes, dtype=np.int16)
        return None

    def _streaming_loop(self):
        start_time = time.monotonic()
        duration = self.streaming_config.get('duration', 0)

        try:
            while self.is_streaming:
                # Check for sine wave duration limit
                if self.streaming_config['mode'] == "Sine Wave" and duration > 0:
                    if time.monotonic() - start_time >= duration:
                        break

                if self.client_conn is None: break
                
                signal_chunk = self._get_next_chunk()

                # If get_next_chunk returns None (e.g., end of audio file), stop streaming
                if signal_chunk is None:
                    break

                # If chunk is smaller than expected (end of file), pad with zeros
                if len(signal_chunk) < self.chunk_size:
                    padding = np.zeros(self.chunk_size - len(signal_chunk), dtype=np.int16)
                    signal_chunk = np.concatenate((signal_chunk, padding))

                self.client_conn.sendall(signal_chunk.tobytes())
                
                # Update progress for audio file
                if self.streaming_config['mode'] == "Audio File" and self.audio_data:
                    percent = min(100, (self.audio_pos / len(self.audio_data)) * 100)
                    self.progress_callback(percent)

                fft_bytes = self.client_conn.recv(self.chunk_size * 4, socket.MSG_WAITALL)
                if not fft_bytes:
                    self.status_callback("Client disconnected.", "orange")
                    break
                
                fft_data = np.frombuffer(fft_bytes, dtype=np.float32)
                self.data_queue.put((signal_chunk, fft_data, self.rate))
                time.sleep(self.chunk_size / self.rate * 0.9)

        except (socket.error, ConnectionResetError, BrokenPipeError) as e:
            print(f"Network error during streaming: {e}")
            if self.is_server_running:
                self.status_callback("Client disconnected.", "orange")
        except Exception as e:
            print(f"An unexpected error occurred in the streaming loop: {e}")
        finally:
            self.is_streaming = False
            self.stream_stopped_callback()