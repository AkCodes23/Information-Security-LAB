# Lab 2: DES, 3DES, AES (128/192/256), modes (ECB/CBC/CTR), benchmarking
# Requires: pip install pycryptodome
# Run: python lab2_symmetric.py

import time
from Crypto.Cipher import DES, DES3, AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

def to_bytes(s): return s.encode() if isinstance(s, str) else s

def des_ecb_encrypt(msg, key8):
    cipher = DES.new(to_bytes(key8)[:8], DES.MODE_ECB)
    ct = cipher.encrypt(pad(to_bytes(msg), 8))
    return ct

def des_ecb_decrypt(ct, key8):
    cipher = DES.new(to_bytes(key8)[:8], DES.MODE_ECB)
    return unpad(cipher.decrypt(ct), 8)

def des_cbc_encrypt(msg, key8, iv8):
    cipher = DES.new(to_bytes(key8)[:8], DES.MODE_CBC, iv=to_bytes(iv8)[:8])
    return cipher.encrypt(pad(to_bytes(msg), 8))

def des_cbc_decrypt(ct, key8, iv8):
    cipher = DES.new(to_bytes(key8)[:8], DES.MODE_CBC, iv=to_bytes(iv8)[:8])
    return unpad(cipher.decrypt(ct), 8)

def des3_ecb_encrypt(msg, key24):
    key24 = to_bytes(key24)[:24]
    cipher = DES3.new(key24, DES3.MODE_ECB)
    return cipher.encrypt(pad(to_bytes(msg), 8))

def des3_ecb_decrypt(ct, key24):
    key24 = to_bytes(key24)[:24]
    cipher = DES3.new(key24, DES3.MODE_ECB)
    return unpad(cipher.decrypt(ct), 8)

def aes_ecb_encrypt(msg, key):
    key = to_bytes(key)
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(pad(to_bytes(msg), AES.block_size))

def aes_ecb_decrypt(ct, key):
    key = to_bytes(key)
    cipher = AES.new(key, AES.MODE_ECB)
    return unpad(cipher.decrypt(ct), AES.block_size)

def aes_cbc_encrypt(msg, key, iv16):
    cipher = AES.new(to_bytes(key), AES.MODE_CBC, iv=to_bytes(iv16)[:16])
    return cipher.encrypt(pad(to_bytes(msg), 16))

def aes_cbc_decrypt(ct, key, iv16):
    cipher = AES.new(to_bytes(key), AES.MODE_CBC, iv=to_bytes(iv16)[:16])
    return unpad(cipher.decrypt(ct), 16)

def aes_ctr_encrypt(msg, key, nonce8=b'\x00'*8):
    cipher = AES.new(to_bytes(key), AES.MODE_CTR, nonce=to_bytes(nonce8)[:8])
    return cipher.encrypt(to_bytes(msg)), cipher.nonce

def aes_ctr_decrypt(ct, key, nonce8):
    cipher = AES.new(to_bytes(key), AES.MODE_CTR, nonce=to_bytes(nonce8)[:8])
    return cipher.decrypt(ct)

def benchmark(name, enc, dec, msg):
    t0 = time.perf_counter(); c = enc(msg); t1 = time.perf_counter()
    _ = dec(c); t2 = time.perf_counter()
    return (name, t1-t0, t2-t1)

if __name__ == "__main__":
    # 1) DES with key "A1B2C3D4"
    des_key = "A1B2C3D4"
    msg1 = "Confidential Data"
    c1 = des_ecb_encrypt(msg1, des_key)
    print("DES ECB ct:", c1.hex(), "->", des_ecb_decrypt(c1, des_key).decode())

    # 2) AES-128 with key 32 hex chars (16 bytes)
    aes128_key = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF")
    msg2 = "Sensitive Information"
    c2 = aes_ecb_encrypt(msg2, aes128_key)
    print("AES-128 ECB ct:", c2.hex(), "->", aes_ecb_decrypt(c2, aes128_key).decode())

    # 3) Compare DES vs AES-256 times
    test_msg = "Performance Testing of Encryption Algorithms" * 1000
    aes256_key = get_random_bytes(32)
    des_time = benchmark("DES-ECB", lambda m: des_ecb_encrypt(m, des_key), lambda c: des_ecb_decrypt(c, des_key), test_msg)
    aes256_time = benchmark("AES-256-ECB", lambda m: aes_ecb_encrypt(m, aes256_key), lambda c: aes_ecb_decrypt(c, aes256_key), test_msg)
    print("Benchmark:", des_time, aes256_time)

    # 4) 3DES
    des3_key = bytes.fromhex("1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF")
    msg3 = "Classified Text"
    c3 = des3_ecb_encrypt(msg3, des3_key)
    print("3DES ECB ct:", c3.hex(), "->", des3_ecb_decrypt(c3, des3_key).decode())

    # 5) AES-192 with step prints (high-level)
    aes192_key = bytes.fromhex("FEDCBA9876543210FEDCBA9876543210")[:24]
    msg4 = "Top Secret Data"
    c4 = aes_ecb_encrypt(msg4, aes192_key)
    print("AES-192 ECB ct:", c4.hex(), "->", aes_ecb_decrypt(c4, aes192_key).decode())

    # Additional: CBC and CTR demos
    iv = b"12345678"  # for DES CBC
    c_cbc = des_cbc_encrypt("Secure Communication", des_key, iv)
    print("DES-CBC ct:", c_cbc.hex(), "->", des_cbc_decrypt(c_cbc, des_key, iv).decode())

    nonce = b"\x00"*8
    ctr_ct, used_nonce = aes_ctr_encrypt("Cryptography Lab Exercise", aes128_key, nonce)
    print("AES-CTR ct:", ctr_ct.hex(), "->", aes_ctr_decrypt(ctr_ct, aes128_key, used_nonce).decode())
