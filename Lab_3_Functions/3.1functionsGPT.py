import subprocess
import sys

# Check if the cryptography library is installed
try:
    import cryptography
    from cryptography.fernet import Fernet

    # Generate and test encryption to confirm the library works
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    message = input("Input a string to encrypt:")
    encrypted_message = cipher_suite.encrypt(message)
    decrypted_message = cipher_suite.decrypt(encrypted_message)

    print("Cryptography library is installed and working!")
    print(f"Original message: {message}")
    print(f"Encrypted message: {encrypted_message}")
    print(f"Decrypted message: {decrypted_message}")
except ImportError:
    print("Cryptography library is not installed.")
    print("Installing cryptography...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography"])
    print("Cryptography library installed. Run the program again.")
