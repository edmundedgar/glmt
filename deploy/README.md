# Dev-phase deployment: reverse tunnel + nginx relay

Temporary setup so `labels.goat.navy` (cloud host `o1.edochan.com`) can serve
real `queryLabels`/`subscribeLabels` traffic while the Ingester, Label DB, and
Labeler Server all still run on the local GPU box. Sidesteps building the
local↔cloud DB sync mechanism until that's actually needed — see
`docs/NOTES.md` for that design discussion. This setup is not throwaway: it's
structurally identical to the eventual production architecture (nginx
terminating TLS, reverse-proxying to the labeler server); switching off the
tunnel later is just changing nginx's upstream to a same-box port.

## Outstanding blocker: SSH key auth isn't working yet

`ssh gmlt@o1.edochan.com` currently fails with `Permission denied
(publickey)`. The local machine is offering this key:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIiChjw3/qGgpnXrQSgqeODQHl9oEHq1jqUCu9Qoo6v3 glmt@yew
```

On `o1.edochan.com`, as `gmlt`, confirm it's actually present and the
permissions are right (a too-open `~/.ssh` or `authorized_keys` makes sshd
silently ignore the file):

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIiChjw3/qGgpnXrQSgqeODQHl9oEHq1jqUCu9Qoo6v3 glmt@yew" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Then re-test from the local box: `ssh gmlt@o1.edochan.com echo ok`.

## Local box (GPU machine) setup

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

The unit runs `autossh -R 127.0.0.1:14831:127.0.0.1:14831 gmlt@o1.edochan.com`
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
sudo cp nginx-labels.goat.navy.conf /etc/nginx/sites-available/labels.goat.navy
sudo ln -s /etc/nginx/sites-available/labels.goat.navy /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

(If this host's nginx doesn't use the `sites-available`/`sites-enabled`
convention, drop the file straight into `/etc/nginx/conf.d/` instead.)

This config is **HTTP-only on purpose**. Once DNS for `labels.goat.navy`
points here and you're ready for TLS:

```bash
sudo certbot --nginx -d labels.goat.navy
```

Certbot's nginx plugin will detect this server block and rewrite it to add
the HTTPS block (with the Let's Encrypt cert paths) plus an HTTP→HTTPS
redirect — no manual edits needed.

## End-to-end test once both sides are up

With nothing else listening on 14831 anywhere, a quick loopback test from
the GPU box confirms the whole chain.

**`python -m http.server` serves the entire current directory, dotfiles
included — running it from the repo root would publish `.env` (the
Anthropic API key) to the internet for as long as it's up.** Always `cd`
to an empty scratch directory first:

```bash
mkdir -p /tmp/labeler-smoketest && cd /tmp/labeler-smoketest
python -m http.server 14831   # stand in for the labeler server, quick smoke test
```

Then, from anywhere:

```bash
curl http://labels.goat.navy/  # should reach the Python server above
```

Swap in the real labeler server once `@skyware/labeler` (Component 4) is
actually built.
