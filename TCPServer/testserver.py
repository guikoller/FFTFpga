import socket
import struct

# Configurações
HOST = '192.168.0.8'   # Aceita conexões de qualquer IP
PORT = 5000        # Porta do servidor
N = 1024           # Tamanho da sequência

# Cria a sequência de 0 até N-1 como inteiros de 16 bits (int16_t)
data = struct.pack(f'{N}h', *range(N))  # 'h' = signed short (int16_t)

# Criação do socket TCP
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"Servidor escutando em {HOST}:{PORT}")

    conn, addr = s.accept()
    with conn:
        print(f"Conexão de {addr}")
        total_sent = 0
        while total_sent < len(data):
            sent = conn.send(data[total_sent:])
            if sent == 0:
                print("Conexão encerrada prematuramente.")
                break
            total_sent += sent
        print("Dados enviados com sucesso.")
