def caesar_decrypt(ciphertext, shift):
    return ''.join(chr((ord(c.lower()) - ord('a') - shift) % 26 + ord('a')) for c in ciphertext if c.isalpha())

def main():
    sample_ct = "CIW"
    sample_pt = "yes"
    shift = (ord(sample_ct[0]) - ord(sample_pt[0])) % 26

    print("Detected Caesar shift:", shift)

    tablet_ct = "XVIEWYWI"
    plaintext = caesar_decrypt(tablet_ct, shift)
    print("Decrypted tablet text:", plaintext)

if __name__ == "__main__":
    main()
