"""Entry point for running Verda Cloud MCP Server - COMPACT EDITION."""

# Use compact server with 21 mega-tools (was 104 separate tools)
from .server_compact import main

if __name__ == "__main__":
    main()
