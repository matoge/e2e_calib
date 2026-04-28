"""Tiny HTTP reverse proxy with Cf-Connecting-Ip allowlist.

Sits between cloudflared and a local service (e.g., ClearML web UI on :18080).
Drops any request whose Cf-Connecting-Ip header is not in the allowlist.

Usage:
    python scripts/mcp/cf_ip_proxy.py \
        --listen 18081 --upstream http://localhost:18080 \
        --allow 103.175.111.222

cloudflared then tunnels 18081 (not 18080) so direct hits without the
Cf-Connecting-Ip header (= not via cloudflare) get 403'd, and CF requests
with non-allowlisted client IP also 403.
"""
from __future__ import annotations

import argparse
import asyncio

import aiohttp
from aiohttp import web


async def proxy(request: web.Request) -> web.StreamResponse:
    allowed: set[str] = request.app['allowed']
    cip = request.headers.get('cf-connecting-ip', '')
    if cip not in allowed:
        return web.Response(text=f'forbidden (cf-ip={cip!r})', status=403)

    upstream = request.app['upstream']
    url = upstream.rstrip('/') + request.rel_url.path_qs

    # forward — keep Host header so the upstream sets cookies / generates URLs
    # using the public domain (otherwise browser receives Set-Cookie domain=
    # 'localhost', login session is rejected, infinite /login redirect loop).
    session: aiohttp.ClientSession = request.app['session']
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in {'content-length'}}
    body = await request.read() if request.method not in ('GET', 'HEAD') else None

    async with session.request(request.method, url, headers=headers, data=body,
                                allow_redirects=False) as up:
        # Pass Content-Encoding through so the client can decompress. Drop only
        # transfer-encoding (chunked is reconstructed by aiohttp) and content-
        # length (recomputed downstream). Previously stripping content-encoding
        # caused the browser to receive raw gzip bytes for ClearML's web UI.
        resp = web.StreamResponse(status=up.status, reason=up.reason,
                                   headers={k: v for k, v in up.headers.items()
                                            if k.lower() not in {'transfer-encoding',
                                                                  'content-length'}})
        await resp.prepare(request)
        async for chunk in up.content.iter_any():
            await resp.write(chunk)
        await resp.write_eof()
        return resp


async def on_startup(app: web.Application):
    app['session'] = aiohttp.ClientSession(auto_decompress=False)


async def on_cleanup(app: web.Application):
    await app['session'].close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--listen', type=int, required=True, help='local port to bind')
    ap.add_argument('--upstream', required=True, help='http://host:port to forward to')
    ap.add_argument('--allow', action='append', default=[],
                    help='allow this Cf-Connecting-Ip (repeatable)')
    ap.add_argument('--bind', default='127.0.0.1')
    args = ap.parse_args()

    if not args.allow:
        raise SystemExit('--allow IP must be set; refuse to start an open proxy')

    app = web.Application(client_max_size=1024 ** 3)  # 1 GB
    app['allowed'] = set(args.allow)
    app['upstream'] = args.upstream
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route('*', '/{path:.*}', proxy)
    print(f'[cf_ip_proxy] {args.bind}:{args.listen} → {args.upstream}'
          f'   allow cf-ip in {sorted(app["allowed"])}')
    web.run_app(app, host=args.bind, port=args.listen, print=None)


if __name__ == '__main__':
    main()
