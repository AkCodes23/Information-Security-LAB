def caesar_decrypt(ciphertext, key):
    result = ''
    for c in ciphertext:
        if c.isalpha():
            base = ord('A') if c.isupper() else ord('a')
            result += chr((ord(c) - base - key) % 26 + base)
        else:
            result += c
    return result

def main():
    ciphertext = "NCJAEZRCLAS/LYODEPRLYZRCLASJLCPEHZDTOPDZOLN&BY"
    for key in range(10, 17):  # Keys close to 13
        plain = caesar_decrypt(ciphertext, key)
        print(f"Key {key}: {plain}")

if __name__ == "__main__":
    main()
