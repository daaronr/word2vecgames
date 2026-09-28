#!/usr/bin/env python3
"""
Simple HTTP server to serve the Word Bocce frontend.
Runs on port 8080 by default.
"""

import http.server
import socketserver
import argparse


class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that disables caching for easier development."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description='Serve Word Bocce frontend')
    parser.add_argument('--port', type=int, default=8080, help='Port to serve on (default: 8080)')
    args = parser.parse_args()

    PORT = args.port

    with socketserver.TCPServer(("", PORT), NoCacheHTTPRequestHandler) as httpd:
        print(f"✓ Frontend server running at http://localhost:{PORT}")
        print(f"✓ Open http://localhost:{PORT} in your browser")
        print(f"\nMake sure the backend is running on http://localhost:8000")
        print("Press Ctrl+C to stop\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down...")


if __name__ == "__main__":
    main()
