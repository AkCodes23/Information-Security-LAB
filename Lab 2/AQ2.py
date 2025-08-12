from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad


key = bytes.fromhex("A1B2C3D4E5F60708")
block1 = bytes.fromhex("54686973206973206120636f6e666964656e7469616c206d657373616765")
block2 = bytes.fromhex("416e64207468697320697320746865207365636f6e6420626c6f636b")


cipher = DES.new(key, DES.MODE_ECB)

ciphertext1 = cipher.encrypt(pad(block1, DES.block_size))
ciphertext2 = cipher.encrypt(pad(block2, DES.block_size))

print("Ciphertext Block 1:", ciphertext1.hex())
print("Ciphertext Block 2:", ciphertext2.hex())

decrypted1 = unpad(cipher.decrypt(ciphertext1), DES.block_size)
decrypted2 = unpad(cipher.decrypt(ciphertext2), DES.block_size)

print("Decrypted Block 1:", decrypted1.decode())
print("Decrypted Block 2:", decrypted2.decode())

