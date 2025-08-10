import socket
import pdb


def remote_pdb(port: int | None = None, host: str = "127.0.0.1") -> pdb.Pdb:
    """
    Create a remote PDB debugger that accepts connections over a socket.

    Returns a debugger instance that you can use to start debugging at any point
    in your code. The debugger will listen for connections on the specified port.

    This is particularly useful for debugging subprocesses.

    Parameters
    ----------
    port : int | None, optional
        Port to listen on. If None, an available port will be automatically chosen.
    host : str, optional
        Host address to bind to, by default "127.0.0.1"

    Returns
    -------
    pdb.Pdb
        A debugger instance ready to accept remote connections

    Examples
    --------
    One-liner usage:
    >>> from src.debug import remote_pdb
    >>> remote_pdb(4444).set_trace()

    Multi-line usage:
    >>> debugger = remote_pdb(4444)
    >>> debugger.set_trace()  # Start debugging here
    >>> debugger.close()  # Clean up when done

    Connection:
    Connect from another terminal using netcat:
    $ nc 127.0.0.1 4444

    Then use standard PDB commands like 'l' (list), 'p variable' (print),
    'n' (next), 's' (step), 'u' (up stack), 'd' (down stack), 'c' (continue)
    """
    print(f"🐛 Starting socket pdb on {host}:{port if port else 'auto'}")

    class SocketPdb(pdb.Pdb):
        def __init__(self, host, port):
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            if port is None:
                self.socket.bind((host, 0))
                self.actual_port = self.socket.getsockname()[1]
            else:
                self.socket.bind((host, port))
                self.actual_port = port

            print(
                f"🐛 Socket pdb listening on {host}:{self.actual_port}; Connect with `nc {host} {self.actual_port}`"
            )
            print(f"🔗 Connect with: nc {host} {self.actual_port}")
            print("🔗 Waiting for connection... ")
            self.socket.listen(1)
            self.conn, addr = self.socket.accept()
            print(f"✅ Connected from {addr}")

            self.file = self.conn.makefile("rw")
            pdb.Pdb.__init__(self, stdin=self.file, stdout=self.file)

        def close(self):
            try:
                self.file.close()
                self.conn.close()
                self.socket.close()
                print("🔌 Debug session closed")
            except Exception:
                pass

    return SocketPdb(host, port)
