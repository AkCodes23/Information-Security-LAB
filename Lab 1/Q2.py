def vigenere_cipher(text, key, mode='encrypt'):
    text, key = text.lower(), key.lower()
    result = ''
    for i in range(len(text)):
        if text[i].isalpha():
            shift = ord(key[i % len(key)]) - ord('a')
            if mode == 'encrypt':
                result += chr((ord(text[i]) - ord('a') + shift) % 26 + ord('a'))
            else:
                result += chr((ord(text[i]) - ord('a') - shift) % 26 + ord('a'))
    return result


def autokey_cipher_encrypt(text, key):
    text = text.lower()
    key_stream = [key] + [ord(c) - ord('a') for c in text]
    result = ''
    for i in range(len(text)):
        shift = key_stream[i]
        result += chr((ord(text[i]) - ord('a') + shift) % 26 + ord('a'))
    return result


def autokey_cipher_decrypt(cipher, key):
    cipher = cipher.lower()
    result = ''
    for i in range(len(cipher)):
        shift = key if i == 0 else ord(result[i - 1]) - ord('a')
        result += chr((ord(cipher[i]) - ord('a') - shift) % 26 + ord('a'))
    return result


def main():
    message = input("Enter Message:").replace(" ", "")

    key_vigenere =input("Enter Key:")
    encrypted_vigenere = vigenere_cipher(message, key_vigenere)
    decrypted_vigenere = vigenere_cipher(encrypted_vigenere, key_vigenere, mode='decrypt')
  
    key_autokey = int(input("Enter Key:"))
    encrypted_autokey = autokey_cipher_encrypt(message, key_autokey)
    decrypted_autokey = autokey_cipher_decrypt(encrypted_autokey, key_autokey)

    print("\n📘 Vigenère Cipher")
    print("Encrypted:", encrypted_vigenere)
    print("Decrypted:", decrypted_vigenere)

    print("\n📗 Autokey Cipher")
    print("Encrypted:", encrypted_autokey)
    print("Decrypted:", decrypted_autokey)


if __name__ == "__main__":
    main()
