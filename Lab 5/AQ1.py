import socket
import threading
import hashlib
import time

HOST = '127.0.0.1'
PORT = 65432

# -------------------------
# Hash function
# -------------------------
def compute_hash(message: str) -> str:
    return hashlib.sha256(message.encode()).hexdigest()

# -------------------------
# Server function
# -------------------------
def server_program():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[SERVER] Listening on {HOST}:{PORT}...")

        conn, addr = s.accept()
        with conn:
            print(f"[SERVER] Connected by {addr}")
            full_message = ""

            while True:
                chunk = conn.recv(1024).decode()
                if chunk == "END":  # End of message signal
                    break
                full_message += chunk

            print(f"[SERVER] Full message received: {full_message}")
            message_hash = compute_hash(full_message)
            print(f"[SERVER] Computed hash: {message_hash}")

            conn.sendall(message_hash.encode())

# -------------------------
# Client function
# -------------------------
def client_program(message):
    time.sleep(1)  # Give server time to start
    chunks = [message[i:i+10] for i in range(0, len(message), 10)]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        for chunk in chunks:
            s.sendall(chunk.encode())
            time.sleep(0.1)  # Simulate delay between chunks

        s.sendall("END".encode())  # Signal end of message

        server_hash = s.recv(1024).decode()
        local_hash = compute_hash(message)

        print(f"[CLIENT] Original message: {message}")
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
    # Change this to simulate tampering
    message_to_send = "This is a long message sent in multiple parts to test integrity."
    # message_to_send = "This is a tampered message!"  # Uncomment to simulate corruption

    # Start server in background
    server_thread = threading.Thread(target=server_program, daemon=True)
    server_thread.start()

    # Run client
    client_program(message_to_send)
