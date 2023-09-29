import aleo_python_sdk as aleo

if __name__ == "__main__":
    private_key = aleo.PrivateKey()
    print(private_key.to_string())