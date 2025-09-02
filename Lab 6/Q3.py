# client_server_dh_hmac_demo.py
# Single script: server and client threads. DH key exchange -> XOR encryption + HMAC-SHA256 integrity.

import socket
import threading
import time
import secrets
import hashlib

HOST = "127.0.0.1"
PORT = 65439

# ---- Minimal prime/gen (educational) ----
def is_probable_prime(n, k=12):
    if n < 2: return False
    small = [2,3,5,7,11,13,17,19,23,29]
    for p in small:
        if n == p: return True
        if n % p == 0: return False
    d = n-1; s = 0
    while d % 2 == 0:
        s += 1; d //= 2
    for _ in range(k):
        a = secrets.randbelow(n-3)+2
        x = pow(a, d, n)
        if x in (1, n-1): continue
        for __ in range(s-1):
            x = (x*x) % n
            if x == n-1: break
        else:
            return False
    return True

def gen_prime(bits=256):
    while True:
        n = secrets.randbits(bits) | 1 | (1 << (bits-1))
        if is_probable_prime(n):
            return n

def find_generator(p):
    while True:
        g = secrets.randbelow(p-3) + 2
        if pow(g, 2, p) != 1 and pow(g, (p-1)//2, p) != 1:
            return g

def kdf(shared_int: int, out_len: int) -> bytes:
    out = bytearray()
    counter = 0
    seed = shared_int.to_bytes((shared_int.bit_length()+7)//8 or 1, "big")
    while len(out) < out_len:
        out.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    return bytes(out[:out_len])

def xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))

def hmac_sha256(key: bytes, data: bytes) -> bytes:
    block = 64
    if len(key) > block:
        key = hashlib.sha256(key).digest()
    key = key.ljust(block, b"\x00")
    o_key_pad = bytes(b ^ 0x5c for b in key)
    i_key_pad = bytes(b ^ 0x36 for b in key)
    return hashlib.sha256(o_key_pad + hashlib.sha256(i_key_pad + data).digest()).digest()

# ---- Server ----
def server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[SERVER] Listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"[SERVER] Connected from {addr}")

            # Receive p, g, A
            p = int.from_bytes(conn.recv(4), "big")
            p = int.from_bytes(conn.recv(p), "big")
            g = int.from_bytes(conn.recv(4), "big")
            g = int.from_bytes(conn.recv(g), "big")
            A = int.from_bytes(conn.recv(4), "big")
            A = int.from_bytes(conn.recv(A), "big")

            # Server secret and public
            b = secrets.randbelow(p-2)+1
            B = pow(g, b, p)

            # Send B
            B_bytes = B.to_bytes((B.bit_length()+7)//8 or 1, "big")
            conn.sendall(len(B_bytes).to_bytes(4, "big") + B_bytes)

            # Shared secret
            s_shared = pow(A, b, p)
            key_enc = kdf(s_shared, 32)
            key_mac = kdf(s_shared + 1, 32)

            # Receive msg: len || ct || hmac
            n = int.from_bytes(conn.recv(4), "big")
            ct = conn.recv(n)
            tag = conn.recv(32)

            # Verify HMAC over ct
            if hmac_sha256(key_mac, ct) != tag:
                print("[SERVER] HMAC verification failed. Tampered or corrupted.")
                return

            # Decrypt (generate keystream from key_enc)
            ks = kdf(int.from_bytes(key_enc, "big"), len(ct))
            pt = xor_bytes(ct, ks)
            print("[SERVER] Received (decrypted):", pt.decode())

# ---- Client ----
def client(message: str):
    time.sleep(0.6)  # let server start
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        # Group and key
        p = gen_prime(256)
        g = find_generator(p)
        a = secrets.randbelow(p-2)+1
        A = pow(g, a, p)

        # Send p, g, A (length-prefixed big-endian)
        def send_int(val: int):
            b = val.to_bytes((val.bit_length()+7)//8 or 1, "big")
            s.sendall(len(b).to_bytes(4, "big") + b)

        send_int(p); send_int(g); send_int(A)

        # Receive B
        m = int.from_bytes(s.recv(4), "big")
        B = int.from_bytes(s.recv(m), "big")

        # Shared secret
        s_shared = pow(B, a, p)
        key_enc = kdf(s_shared, 32)
        key_mac = kdf(s_shared + 1, 32)

        # Encrypt + HMAC
        pt = message.encode()
        ks = kdf(int.from_bytes(key_enc, "big"), len(pt))
        ct = xor_bytes(pt, ks)
        tag = hmac_sha256(key_mac, ct)

        s.sendall(len(ct).to_bytes(4, "big") + ct + tag)

        print("[CLIENT] Sent ciphertext (hex):", ct.hex())
        print("[CLIENT] HMAC (hex):", tag.hex())

if __name__ == "__main__":
    msg = "Client–Server DH + HMAC demo: confidentiality + integrity."
    t = threading.Thread(target=server, daemon=True)
    t.start()
    client(msg)
    time.sleep(1.0)
    print("[MAIN] Done.")
