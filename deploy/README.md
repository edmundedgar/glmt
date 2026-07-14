# Dev-phase deployment: reverse tunnel + nginx relay

Temporary setup so `label.goat.navy` (cloud host `o1.edochan.com`) can serve
real `queryLabels`/`subscribeLabels` traffic while the Ingester, Label DB, and
Labeler Server all still run on the local GPU box. Sidesteps building the
local↔cloud DB sync mechanism until that's actually needed — see
`docs/NOTES.md` for that design discussion. This setup is not throwaway: it's
structurally identical to the eventual production architecture (nginx
terminating TLS, reverse-proxying to the labeler server); switching off the
tunnel later is just changing nginx's upstream to a same-box port.

## Local box (GPU machine) setup

Remote user on `o1.edochan.com` is `glmt` (not `gmlt` — an earlier typo in
this doc pointed at the wrong username and cost some debugging time; the
key/account setup itself was fine).

```bash
sudo apt-get install -y autossh
sudo cp deploy/labeler-tunnel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now labeler-tunnel.service
```

Check it's actually connected:

```bash
sudo systemctl status labeler-tunnel
journalctl -u labeler-tunnel -f
```

The unit runs `autossh -R 127.0.0.1:14831:127.0.0.1:14831 glmt@o1.edochan.com`
— binds port 14831 on the cloud host's **loopback only** (so nothing but
nginx on that same box can reach it directly), forwarding to port 14831 on
this machine, where the labeler server (`@skyware/labeler`, per
`docs/spec.md` Component 4) is expected to listen.

If the tunnel fails to establish, check `AllowTcpForwarding` isn't disabled
in `/etc/ssh/sshd_config` on the cloud host — some hardened configs turn
this off by default.

## Cloud host (`o1.edochan.com`) setup

```bash
sudo apt-get install -y nginx
sudo cp nginx-label.goat.navy.conf /etc/nginx/sites-available/label.goat.navy
sudo ln -s /etc/nginx/sites-available/label.goat.navy /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

(If this host's nginx doesn't use the `sites-available`/`sites-enabled`
convention, drop the file straight into `/etc/nginx/conf.d/` instead.)

This config is **HTTP-only on purpose**. Once DNS for `label.goat.navy`
points here and you're ready for TLS:

```bash
sudo certbot --nginx -d label.goat.navy
```

Certbot's nginx plugin will detect this server block and rewrite it to add
the HTTPS block (with the Let's Encrypt cert paths) plus an HTTP→HTTPS
redirect — no manual edits needed.

## Running the real labeler server

Once the tunnel and nginx are both up, start the actual labeler server on
the GPU box — see `docs/RUNBOOK.md`'s "Labeler server" section for setup
(`.env` vars, `npm install`, `declare-labels.mjs`) and the exact run
command. It listens on `127.0.0.1:14831`, which is exactly the port the
tunnel forwards, so no further config is needed on this end.

Quick end-to-end check once it's running:

```bash
curl https://label.goat.navy/xrpc/com.atproto.label.queryLabels?uriPatterns=at://did:plc:example/app.bsky.feed.post/xyz
```

should reach the labeler server and get a real (possibly empty) `queryLabels`
response back, not a connection error.

**One-time bootstrapping note:** before the real server existed, this setup
was smoke-tested with `python -m http.server 14831` standing in for it.
`python -m http.server` serves the entire current directory, dotfiles
included — if you ever repeat a test like this, `cd` to an empty scratch
directory first (e.g. `mkdir -p /tmp/labeler-smoketest && cd
/tmp/labeler-smoketest`) rather than running it from the repo root, or it
will publish `.env` to the internet for as long as it's up. This actually
happened briefly during initial setup; no unauthorized access occurred
(checked via server logs) but it's a sharp edge worth avoiding a second time.
