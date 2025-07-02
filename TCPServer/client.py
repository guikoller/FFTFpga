# client.py
import socket
import numpy as np
import struct

# --- Constants ---
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server
CHUNK_SAMPLES = 2048 # Number of samples for FFT (e.g., 2048)
BYTES_PER_SAMPLE = 2 # 16-bit audio
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE

class FPGASimulator:
    """
    Simulates an FPGA performing FFT.
    This models the data flow:
    TCP -> Input Register -> BRAM -> FFT Core -> BRAM -> Output Register -> TCP
    """
    def __init__(self, num_samples):
        # --- Hardware Component Simulation ---
        
        # 32-bit Input Register (can hold two 16-bit samples)
        self.input_register = np.zeros(2, dtype=np.int16)
        
        # 32-bit Output Register (can hold one 32-bit float magnitude)
        self.output_register = np.zeros(1, dtype=np.float32)

        # On-chip memory (Block RAM) to store the full data chunk
        self.input_bram = np.zeros(num_samples, dtype=np.int16)
        self.output_bram = np.zeros(num_samples, dtype=np.float32)

        print("-> FPGA Simulator initialized.")
        print(f"-> BRAM size: {num_samples} samples.")

    def control_unit_receive_and_load(self, conn):
        """
        Simulates the FSM controlling data reception.
        It reads from TCP, fills the input register, and transfers to BRAM.
        """
        try:
            # In hardware, you'd receive data byte-by-byte or word-by-word.
            # Here, we get the whole chunk for efficiency, then simulate the loading.
            full_chunk_bytes = conn.recv(CHUNK_BYTES, socket.MSG_WAITALL)
            if not full_chunk_bytes: 
                return False, 0 # Return False and 0 bytes if connection is closed

            # Convert the entire byte stream to samples at once
            all_samples = np.frombuffer(full_chunk_bytes, dtype=np.int16)
            
            # Simulate loading into BRAM (in a real FPGA, this would be a loop)
            self.input_bram[:] = all_samples
            
            return True, len(full_chunk_bytes) # Return True and number of bytes
            
        except Exception as e:
            print(f"[CLIENT] [ERROR] Failed to receive data: {e}")
            return False, 0

    def fft_core_process(self):
        """
        Simulates the dedicated FFT hardware core.
        It reads from input BRAM, calculates, and writes to output BRAM.
        """
        # Apply a Hanning window to smooth the signal edges
        windowed_data = self.input_bram * np.hanning(len(self.input_bram))
        
        # The core FFT calculation
        fft_complex = np.fft.fft(windowed_data)
        
        # The magnitude calculation (e.g., using a CORDIC block in hardware)
        fft_magnitude = np.abs(fft_complex) / len(self.input_bram)
        
        # Store results in the output memory
        self.output_bram[:] = fft_magnitude.astype(np.float32)

    def control_unit_send_results(self, conn):
        """
        Simulates the FSM controlling data transmission.
        It reads from output BRAM, fills the output register, and sends via TCP.
        """
        try:
            # For simulation efficiency, we send the whole BRAM content at once.
            # In hardware, a loop would read each value into the output register
            # and signal the TCP stack to send it.
            output_bytes = self.output_bram.tobytes()
            conn.sendall(output_bytes)
            return True, len(output_bytes)
        except Exception as e:
            print(f"[CLIENT] [ERROR] Failed to send FFT data: {e}")
            return False, 0

def main():
    """The client's main execution function."""
    print("FPGA Client Simulator started.")
    
    server_ip = input(f"Enter server IP address (default: {HOST}): ") or HOST
    server_port = input(f"Enter server port (default: {PORT}): ") or PORT

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"[CLIENT] Attempting to connect to server at {server_ip}:{server_port}...")
            s.connect((server_ip, int(server_port)))
            print("[CLIENT] ✅ Connection Established. Awaiting data...")
            
            fpga = FPGASimulator(CHUNK_SAMPLES)
            
            print("[CLIENT] Entering main processing loop...")
            cycle_count = 0
            while True:
                cycle_count += 1
                print(f"\n[CLIENT] ----- Cycle {cycle_count} -----")
                
                # 1. Receive Data
                print("[CLIENT] 📥 Waiting to receive data chunk from server...")
                success, bytes_received = fpga.control_unit_receive_and_load(s)
                if not success:
                    print("[CLIENT] Server closed the connection. Exiting loop.")
                    break
                print(f"[CLIENT]    -> OK. Received {bytes_received} bytes.")

                # 2. Process FFT
                print("[CLIENT] ⚙️  Starting FFT processing...")
                fpga.fft_core_process()
                print("[CLIENT]    -> OK. FFT processing finished.")

                # 3. Send Results
                print("[CLIENT] 📤 Preparing to send results...")
                success, bytes_sent = fpga.control_unit_send_results(s)
                if not success:
                    print("[CLIENT] Failed to send results. Exiting loop.")
                    break
                print(f"[CLIENT]    -> OK. Sent {bytes_sent} bytes.")
                print("[CLIENT] ----- Cycle Complete -----")


    except ConnectionRefusedError:
        print("[CLIENT] [ERROR] Connection refused. Is the server running and listening on the correct IP/Port?")
    except (socket.timeout, TimeoutError):
        print("[CLIENT] [ERROR] Connection timed out.")
    except Exception as e:
        print(f"[CLIENT] [ERROR] An unexpected error occurred: {e}")
    finally:
        print("[CLIENT] Client simulator finished.")

if __name__ == "__main__":
    main()