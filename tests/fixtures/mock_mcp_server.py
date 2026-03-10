# tests/fixtures/mock_mcp_server.py
"""
Minimal MCP server for integration tests.

Run with:  python tests/fixtures/mock_mcp_server.py
Uses stdio transport (stdin/stdout).
"""
from mcp.server.fastmcp import FastMCP

server = FastMCP("test-server")


@server.tool()
def echo(message: str) -> str:
    """Echo back the message."""
    return message


@server.tool()
def add(a: int, b: int) -> str:
    """Add two numbers and return the result."""
    return str(a + b)


@server.tool()
def lookup_customer(customer_id: str) -> str:
    """Look up a customer by their ID. Returns account details as JSON."""
    import json

    return json.dumps(
        {
            "customer_id": customer_id,
            "account_status": "active",
            "kyc_status": "verified",
            "customer_found": True,
        }
    )


if __name__ == "__main__":
    server.run(transport="stdio")
