import sys
import socket
import time

domain = sys.argv[1]
while True:
    try:
        ip_address = socket.gethostbyname(domain)
        print(ip_address)
        break
    except socket.error as e:
        sys.stderr.write("Error: %s\n" % e)
        time.sleep(5)