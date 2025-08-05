import time
from Crypto.Cipher import DES, AES
from Crypto.Util.Padding import pad, unpad

message = b"Performance Testing of Encryption Algorithms"

# DES setup
des_key = b"12345678"
des_cipher = DES.new(des_key, DES.MODE_ECB)
padded_des = pad(message, DES.block_size)

# AES-256 setup
aes_key = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF")
aes_cipher = AES.new(aes_key, AES.MODE_ECB)
padded_aes = pad(message, AES.block_size)

# DES timing
start_des_enc = time.perf_counter()
des_encrypted = des_cipher.encrypt(padded_des)
end_des_enc = time.perf_counter()

start_des_dec = time.perf_counter()
des_decrypted = unpad(des_cipher.decrypt(des_encrypted), DES.block_size)
end_des_dec = time.perf_counter()

# AES timing
start_aes_enc = time.perf_counter()
aes_encrypted = aes_cipher.encrypt(padded_aes)
end_aes_enc = time.perf_counter()

start_aes_dec = time.perf_counter()
aes_decrypted = unpad(aes_cipher.decrypt(aes_encrypted), AES.block_size)
end_aes_dec = time.perf_counter()

# Results
print(f"DES Encryption Time: {(end_des_enc - start_des_enc)*1e6:.2f} µs")
print(f"DES Decryption Time: {(end_des_dec - start_des_dec)*1e6:.2f} µs")
print(f"AES-256 Encryption Time: {(end_aes_enc - start_aes_enc)*1e6:.2f} µs")
print(f"AES-256 Decryption Time: {(end_aes_dec - start_aes_dec)*1e6:.2f} µs")
