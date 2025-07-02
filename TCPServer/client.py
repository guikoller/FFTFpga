# client.py
import socket
import numpy as np
import struct

# --- Constantes (devem ser as mesmas do servidor) ---
HOST = '127.0.0.1'  # Endereço IP do servidor
PORT = 65432        # Porta do servidor
CHUNK = 1024 * 2    # Tamanho do bloco de áudio (samples)

class HardwareSimulator:
    """
    Simula um hardware dedicado para processamento de FFT.
    Possui registradores de entrada e saída e um módulo de FFT.
    """
    def __init__(self, chunk_size):
        # Registrador para recebimento de dados (simulado por um array numpy)
        self.input_register = np.zeros(chunk_size, dtype=np.int16)
        # Registrador para a saída de dados (FFT em ponto flutuante)
        self.output_register = np.zeros(chunk_size, dtype=np.float32)
        print("-> Módulo de hardware simulado inicializado.")

    def receive_and_load_data(self, conn):
        """Recebe dados do socket e carrega no registrador de entrada."""
        try:
            # Calcula o número de bytes a serem recebidos (2 bytes por sample int16)
            data_bytes = conn.recv(CHUNK * 2, socket.MSG_WAITALL)
            if not data_bytes:
                return False
            
            self.input_register = np.frombuffer(data_bytes, dtype=np.int16)
            return True
        except Exception as e:
            print(f"[ERRO] Falha ao receber dados: {e}")
            return False

    def process_fft(self):
        """
        Módulo da FFT. Pega os dados do input_register, calcula a FFT
        e armazena a magnitude no output_register.
        """
        # Aplica uma janela de Hanning para suavizar as bordas do sinal
        windowed_data = self.input_register * np.hanning(len(self.input_register))
        
        # Calcula a FFT. O resultado é um array de números complexos.
        fft_complex = np.fft.fft(windowed_data)
        
        # Calcula a magnitude (valor absoluto) e normaliza.
        fft_magnitude = np.abs(fft_complex) / CHUNK
        
        # Carrega o resultado no registrador de saída
        self.output_register = fft_magnitude.astype(np.float32)

    def send_output_data(self, conn):
        """Envia os dados do registrador de saída para o servidor."""
        try:
            conn.sendall(self.output_register.tobytes())
        except Exception as e:
            print(f"[ERRO] Falha ao enviar dados da FFT: {e}")

def main():
    """Função principal do cliente."""
    print("Cliente FFT inicializado.")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"Tentando conectar ao servidor em {HOST}:{PORT}...")
            s.connect((HOST, PORT))
            print("Conectado ao servidor.")

            hardware = HardwareSimulator(CHUNK)

            while True:
                # 1. Módulo de recebimento de pacotes
                if not hardware.receive_and_load_data(s):
                    print("O servidor parece ter fechado a conexão.")
                    break
                
                # 2. Módulo da FFT
                hardware.process_fft()
                
                # 3. Módulo de envio de dados
                hardware.send_output_data(s)

    except ConnectionRefusedError:
        print("[ERRO] Conexão recusada. O servidor está desligado ou ocupado?")
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro inesperado: {e}")
    finally:
        print("Cliente encerrado.")

if __name__ == "__main__":
    main()