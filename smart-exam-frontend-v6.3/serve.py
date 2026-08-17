#!/usr/bin/env python3

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT.parent / ".env")
load_dotenv(ROOT / ".env")

BACKEND_URL = os.getenv(
    "API_URL",
    "http://localhost:8000",
).rstrip("/")

API_KEY = os.getenv(
    "API_KEY",
    "change-this-in-production",
)

FRONTEND_HOST = os.getenv(
    "FRONTEND_HOST",
    "127.0.0.1",
)

FRONTEND_PORT = int(
    os.getenv(
        "FRONTEND_PORT",
        "5500",
    )
)


class FrontendHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            directory=str(ROOT),
            **kwargs,
        )

    def end_headers(self):
        self.send_header(
            "Cache-Control",
            "no-store, no-cache, must-revalidate, max-age=0",
        )
        self.send_header(
            "Pragma",
            "no-cache",
        )
        self.send_header(
            "Expires",
            "0",
        )
        super().end_headers()

    def log_message(self, fmt, *args):
        print(
            "[frontend]",
            fmt % args,
        )

    def do_GET(self):
        if self.path.startswith("/api/"):
            self.proxy_request("GET")
            return

        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self.proxy_request("POST")
            return

        self.send_error(
            404,
            "Not found",
        )

    def proxy_request(self, method):
        parsed = urlsplit(
            self.path
        )

        backend_path = (
            parsed.path[len("/api"):]
        )

        target_url = (
            f"{BACKEND_URL}"
            f"{backend_path}"
        )

        if parsed.query:
            target_url += (
                f"?{parsed.query}"
            )

        headers = {
            "X-API-Key": API_KEY,
        }

        content_type = self.headers.get(
            "Content-Type"
        )

        if content_type:
            headers[
                "Content-Type"
            ] = content_type

        try:
            if method == "GET":
                response = requests.get(
                    target_url,
                    headers=headers,
                    timeout=15,
                )

            else:
                content_length = int(
                    self.headers.get(
                        "Content-Length",
                        "0",
                    )
                )

                body = self.rfile.read(
                    content_length
                )

                response = requests.post(
                    target_url,
                    headers=headers,
                    data=body,
                    timeout=190,
                )

            self.send_response(
                response.status_code
            )

            response_content_type = (
                response.headers.get(
                    "Content-Type",
                    "application/octet-stream",
                )
            )

            self.send_header(
                "Content-Type",
                response_content_type,
            )

            self.end_headers()

            self.wfile.write(
                response.content
            )

        except requests.exceptions.Timeout:
            self.send_response(
                504
            )

            self.send_header(
                "Content-Type",
                "application/json",
            )

            self.end_headers()

            self.wfile.write(
                b'{"detail":"Backend request timed out."}'
            )

        except requests.exceptions.RequestException as exc:
            self.send_response(
                502
            )

            self.send_header(
                "Content-Type",
                "application/json",
            )

            self.end_headers()

            message = (
                str(exc)
                .replace("\\", "\\\\")
                .replace('"', '\\"')
            )

            self.wfile.write(
                (
                    '{"detail":"Could not reach frozen backend: '
                    + message
                    + '"}'
                ).encode("utf-8")
            )


def main():
    print()
    print(
        "Smart Exam Evaluation — V6.3 adaptive-results frontend"
    )
    print(
        f"Frontend: http://{FRONTEND_HOST}:{FRONTEND_PORT}"
    )
    print(
        f"Backend proxy target: {BACKEND_URL}"
    )
    print()
    print(
        "Keep your existing FastAPI backend and OCR service running."
    )
    print()

    server = ThreadingHTTPServer(
        (
            FRONTEND_HOST,
            FRONTEND_PORT,
        ),
        FrontendHandler,
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print(
            "\nStopping frontend."
        )

    finally:
        server.server_close()


if __name__ == "__main__":
    main()
