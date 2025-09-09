# Lab 5: Custom hash (djb2-like), socket integrity demo, MD5/SHA1/SHA256 benchmarking
# Run: python lab5_hashing.py
# For socket demo, run server in one terminal and client in another.

import hashlib, random, string, time, socket, threading

def custom_hash(s: str):
    h = 5381
    for ch in s:
        h = ((h << 5) + h) + ord(ch)  # h*33 + c
        h ^= (h >> 13)  # extra mixing
        h &= 0xFFFFFFFF  # 32-bit mask
    return h

def benchmark_hashes(N=100):
    data = ["".join(random.choice(string.ascii_letters) for _ in range(random.randint(10,50))) for _ in range(N)]
    def measure(fn):
        t0 = time.perf_counter()
        hs = [fn(d.encode()).hexdigest() for d in data]
        t1 = time.perf_counter()
        return hs, t1-t0
    m5, t5 = measure(hashlib.md5)
    s1, t1 = measure(hashlib.sha1)
    s256, t256 = measure(hashlib.sha256)
    # collision check (expect none on such small sets)
    def collisions(hs):
        seen, col = set(), 0
        for h in hs:
            if h in seen: col += 1
            seen.add(h)
        return col
    return {"MD5": t5, "SHA1": t1, "SHA256": t256, "collisions": {"MD5": collisions(m5), "SHA1": collisions(s1), "SHA256": collisions(s256)}}

# Socket server/client integrity demo
def server(host="127.0.0.1", port=5001):
    s = socket.socket(); s.bind((host, port)); s.listen(1)
    print(f"[Server] Listening on {host}:{port}")
    conn, _ = s.accept()
    data = conn.recv(1<<20)
    h = hashlib.sha256(data).hexdigest().encode()
    conn.sendall(h)
    conn.close(); s.close()

def client(message: bytes, corrupt=False, host="127.0.0.1", port=5001):
    if corrupt:
        message = bytearray(message); message[0] ^= 0xFF; message = bytes(message)
    s = socket.socket(); s.connect((host, port))
    s.sendall(message)
    h_remote = s.recv(4096).decode()
    h_local = hashlib.sha256(message).hexdigest()
    s.close()
    print("Integrity OK?" , h_remote == h_local)

if __name__ == "__main__":
    print("Custom hash:", custom_hash("Hello World"))
    print("Benchmark:", benchmark_hashes(100))

    # Socket demo (server in thread for convenience)
    t = threading.Thread(target=server, daemon=True)
    t.start()
    time.sleep(0.2)
    client(b"Important data packet", corrupt=False)
    # Corruption case
    t2 = threading.Thread(target=server, daemon=True); t2.start(); time.sleep(0.2)
    client(b"Important data packet", corrupt=True)
