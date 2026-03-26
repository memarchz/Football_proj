from pyngrok import ngrok
import http.server
import socketserver
import threading

# Local server settings
PORT = 5000

def start_local_server():
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Local server running at http://localhost:{PORT}")
        httpd.serve_forever()

# Start local HTTP server in a background thread
server_thread = threading.Thread(target=start_local_server, daemon=True)
server_thread.start()

# Open ngrok tunnel
public_url = ngrok.connect(PORT, "http")

print("ngrok tunnel established!")
print("Public URL:", public_url)

# Keep the script alive
input("Press ENTER to stop the tunnel...\n")

# Cleanup
ngrok.disconnect(public_url)
ngrok.kill()
