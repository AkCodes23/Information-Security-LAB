import socket
import threading
import time


# -------------------------
# Custom hash function
# -------------------------
def custom_hash(input_string: str) -> int:
    hash_value = 5381
    for char in input_string:
        hash_value = (hash_value * 33) + ord(char)
        hash_value ^= (hash_value >> 13)
        hash_value ^= (hash_value << 7)
    return hash_value & 0xFFFFFFFF


# -------------------------
# Server function
# -------------------------
def server_program(host='127.0.0.1', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"[SERVER] Listening on {host}:{port}...")

        conn, addr = s.accept()
        with conn:
            print(f"[SERVER] Connected by {addr}")
            data = conn.recv(1024).decode()
            if data:
                print(f"[SERVER] Received data: {data}")
                server_hash = custom_hash(data)
                print(f"[SERVER] Computed hash: {server_hash}")
                conn.sendall(str(server_hash).encode())


# -------------------------
# Client function
# -------------------------
def client_program(message, host='127.0.0.1', port=65432):
    time.sleep(1)  # Give server time to start
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(message.encode())

        server_hash = int(s.recv(1024).decode())
        local_hash = custom_hash(message)

        print(f"[CLIENT] Data sent: {message}")
        print(f"[CLIENT] Local hash: {local_hash}")
        print(f"[CLIENT] Server hash: {server_hash}")

        if local_hash == server_hash:
            print("[CLIENT] ✅ Data integrity verified — no corruption detected.")
        else:
            print("[CLIENT] ❌ Data corruption or tampering detected!")


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    # Change this message to simulate tampering
    message_to_send = "Hello, this is a test message!"
    # message_to_send = "Hello, this is a tampered message!"  # Uncomment to simulate corruption

    # Start server in a background thread
    server_thread = threading.Thread(target=server_program, daemon=True)
    server_thread.start()

    # Run client
    client_program(message_to_send)
