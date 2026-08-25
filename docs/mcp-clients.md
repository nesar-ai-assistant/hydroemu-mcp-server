# Using this server from any MCP client

The same server plugs into Claude Code, the Claude desktop app, Codex,
Cursor — anything that speaks MCP. That versatility is the point of the
protocol.

## The two facts every config needs

1. **stdio launch command**: `python -m mcp_server --transport stdio`, run so
   that this repo is importable — either the working directory is the repo
   root, or `PYTHONPATH` points at it.
2. **HTTP endpoint**: run `python -m mcp_server --transport streamable-http
   --port 8000` in a terminal; clients connect to
   `http://127.0.0.1:8000/mcp`.

`python` must be the environment's Python with `sepia` and other deps
installed. GUI apps do not inherit your shell, so configs below use the
absolute interpreter path — find yours with `which python`.

## Claude Code

A project config (`.mcp.json`) is checked into this repo. Activate the env,
`cd` here, run `claude` — the `hydroemu` server is available (inspect with
`/mcp`). Or add it yourself:

```bash
claude mcp add hydroemu -- python -m mcp_server --transport stdio   # from this repo root
claude mcp add --transport http hydroemu http://127.0.0.1:8000/mcp  # server already running
```

## Claude desktop app

Settings → Developer → Edit Config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "hydroemu": {
      "command": "/ABS/PATH/TO/envs/hydroemu/bin/python",
      "args": ["-m", "mcp_server", "--transport", "stdio"],
      "env": { "PYTHONPATH": "/ABS/PATH/TO/hydroemu-mcp-server" }
    }
  }
}
```

## Codex CLI

`~/.codex/config.toml`:

```toml
[mcp_servers.hydroemu]
command = "/ABS/PATH/TO/envs/hydroemu/bin/python"
args = ["-m", "mcp_server", "--transport", "stdio"]
env = { PYTHONPATH = "/ABS/PATH/TO/hydroemu-mcp-server" }
```

## Cursor (untested)

`.cursor/mcp.json` in a project, or `~/.cursor/mcp.json` globally — same
JSON shape as the Claude desktop config above.

## Running multiple servers at once

This server coexists with `spectra-mcp-server` or any other MCP server;
give each its own port:

```bash
python -m mcp_server --transport streamable-http --port 8000
python -m mcp_server --transport streamable-http --port 8001
```

(first command from `hydroemu-mcp-server/`, second from another server).
Use **distinct ports** — the automatic port-clearing on startup kills any
leftover `mcp_server` process holding the requested port.

For stdio clients, just add both entries, each with its own `PYTHONPATH`:

```json
{
  "mcpServers": {
    "hydroemu": { "command": "/ABS/.../bin/python",
                  "args": ["-m", "mcp_server", "--transport", "stdio"],
                  "env": { "PYTHONPATH": "/ABS/PATH/TO/hydroemu-mcp-server" } },
    "spectra":  { "command": "/ABS/.../bin/python",
                  "args": ["-m", "mcp_server", "--transport", "stdio"],
                  "env": { "PYTHONPATH": "/ABS/PATH/TO/spectra-mcp-server" } }
  }
}
```

## Hosting beyond localhost

**Security first**: these servers have **no authentication**, and their tools
write files to any `output_dir` path. Expose them through unguessable
temporary URLs for demos, or put real auth in front.

### Quick tunnel — free, no account

```bash
brew install cloudflared
MCP_PUBLIC=1 python -m mcp_server --transport streamable-http --port 8000
cloudflared tunnel --url http://127.0.0.1:8000
```

`cloudflared` prints a random `https://<name>.trycloudflare.com` URL; remote
clients connect to `https://<name>.trycloudflare.com/mcp`. Set `MCP_PUBLIC=1`
to disable DNS-rebinding protection for tunnel serving.
