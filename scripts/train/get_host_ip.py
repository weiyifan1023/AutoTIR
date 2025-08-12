import socket
import subprocess

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0].strip()
    except:
        result = subprocess.run("hostname -I", capture_output=True, text=True, shell=True).stdout
        return result.strip().split()[0]

if __name__ == "__main__":
    print(get_host_ip())
