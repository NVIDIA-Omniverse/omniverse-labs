#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# OVRTX Dev Variant Presenter - server watchdog (Linux / macOS peer of run_server.ps1).
# The ovrtx/ovstream native layer can hard-die (no traceback), and the app can self-restart
# (POST /api/restart -> os._exit(43), e.g. the frontend's dry-pipe escalation); this relaunches
# on abnormal exit so a death costs a page reload + auto session restore, not a dead app.
# Ctrl+C exits the loop.
#
# The child is started in the background and waited on by PID (not through a pipe): an abrupt
# os._exit / native exit then still yields a reliable exit code, and no pipe held open by a
# parent process that captures this script's stdout (a wrapper script, a CI job, any launcher
# that redirects) can wedge the watchdog. Stdout/stderr go to logs/ so a death still leaves
# evidence (split into _out / _err; previous run kept as *.prev.log).
# ASCII only, LF line endings only - a CRLF shebang is not executable on Linux.
set -u

proj="$(cd "$(dirname "$0")" && pwd)"
py="$proj/.venv/bin/python"
log="$proj/server_crashes.log"
logdir="$proj/logs"
mkdir -p "$logdir"
out="$logdir/server_out.log"
err="$logdir/server_err.log"
prev_out="$logdir/server_out.prev.log"
prev_err="$logdir/server_err.prev.log"
url_file="$logdir/server_url.txt"

export PYTHONUNBUFFERED=1   # the crash tail must reach the log before the process dies
# The control port is chosen dynamically (8080 if free, else the next free port, else an
# ephemeral one). The child's stdout is redirected to a log, so it drops the resolved URL
# here for us to print to the console after each launch.
export DEV_VARIANT_PRESENTER_URL_FILE="$url_file"

if [ ! -x "$py" ]; then
    # First run on a fresh clone: build the environment, then fall through to the loop.
    # Kept OUT of the relaunch loop on purpose - a crash must relaunch instantly, not resync.
    echo "No .venv found - running 'uv sync' (first-time setup) ..."
    if ! (cd "$proj" && uv sync); then
        echo "*** uv sync failed (is uv installed? https://docs.astral.sh/uv/) - cannot start. ***"
        exit 1
    fi
    if [ ! -x "$py" ]; then
        echo "*** uv sync did not produce $py - cannot start. ***"
        exit 1
    fi
fi

# The browser viewer needs NVIDIA's StreamSDK WebRTC client, which is NOT redistributed in
# this repo (it is NVIDIA's own software, published in NVIDIA-Omniverse/ovstream). Fetch it
# once from a PINNED upstream commit - never HEAD, so an upstream change cannot silently
# alter or break this. The Python wheels ship no JS, so uv sync cannot provide it.
# Kept OUT of the relaunch loop for the same reason as uv sync: a crash must relaunch
# instantly, not re-verify a 700 KB download.
stream_lib="$proj/web/omniverse-webrtc-streaming-library.js"
stream_lib_dir="$proj/web"
stream_lib_url="https://raw.githubusercontent.com/NVIDIA-Omniverse/ovstream/af7f1f9006d1037a3cc7b8eca73f39a6469b69c2/examples/webrtc_client/omniverse-webrtc-streaming-library.js"
stream_lib_sha="447a74830162b91cb92b0a636f02c0b3e668d835e2a4496f560e31e2b48e5c71"
# Download to a PER-PROCESS temp path and only move it into place once the hash checks out, so
# the real filename never exists in a half-written state: an interrupted download (Ctrl+C, a
# dropped link, a power cut) leaves the installed file untouched. The $$ suffix is what keeps
# two concurrent launches out of each other's way - with a shared ".partial", one run deletes
# the other's in-flight download, or hashes its neighbour's half-written bytes. Each run writes
# its own file, and the rename into place is atomic.
stream_lib_tmp="$stream_lib.partial.$$"

# Print the SHA256 of $1 in lowercase hex. Exit status is the important part:
#   0 = hashed OK   1 = a tool was found but FAILED on this file   2 = no sha256 tool at all
# Those three must stay distinct. Collapsing them into one branch reports "no sha256sum/shasum
# on PATH" even when the tool is present and the FILE is the problem, and lets the script sail
# on to claim a verification it never performed. The last fallback is the project's own
# interpreter, which is guaranteed to exist by the time we get here.
sha256_hex() {
    _f="$1"
    _h=""
    if command -v sha256sum >/dev/null 2>&1; then
        _h="$(sha256sum "$_f" 2>/dev/null)"; _h="${_h%% *}"
    elif command -v shasum >/dev/null 2>&1; then
        _h="$(shasum -a 256 "$_f" 2>/dev/null)"; _h="${_h%% *}"
    elif command -v openssl >/dev/null 2>&1; then
        _h="$(openssl dgst -sha256 "$_f" 2>/dev/null)"; _h="${_h##* }"
    elif command -v python3 >/dev/null 2>&1; then
        _h="$(python3 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$_f" 2>/dev/null)"
    elif [ -x "$py" ]; then
        _h="$("$py" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$_f" 2>/dev/null)"
    else
        return 2
    fi
    [ -n "$_h" ] || return 1
    printf '%s\n' "$_h"
    return 0
}

# Existence is NOT integrity. A 0-byte file, a saved HTML error page, a truncated scp, a
# cloud-sync placeholder or a copy from an older pin all "exist", so a bare `[ ! -f ... ]`
# gate accepts every one of them silently - a permanently broken viewer with no diagnostic
# anywhere. Re-hashing ~700 KB costs a few milliseconds, and this whole block stays OUT of the
# relaunch loop, so a crash still relaunches instantly.
stream_lib_need_fetch=1
if [ -f "$stream_lib" ]; then
    stream_lib_have="$(sha256_hex "$stream_lib")"
    stream_lib_rc=$?   # NOT `|| true` - that would clobber the status we are reading
    if [ "$stream_lib_rc" -eq 2 ]; then
        echo "*** Checksum NOT verified (no sha256 tool available) - refusing to run unverified bytes. ***"
        echo "    Tried: sha256sum, shasum -a 256, openssl dgst -sha256, python3, $py"
        exit 1
    elif [ "$stream_lib_rc" -ne 0 ]; then
        echo "*** Could not compute a SHA256 checksum for web/omniverse-webrtc-streaming-library.js. ***"
        echo "    This is a LOCAL hashing failure, not a corrupt file - the file was left alone."
        echo "    Check its permissions, then relaunch."
        exit 1
    elif [ "$stream_lib_have" = "$stream_lib_sha" ]; then
        stream_lib_need_fetch=0
    else
        echo "web/omniverse-webrtc-streaming-library.js does not match the pinned SHA256 - re-fetching it."
        echo "    expected $stream_lib_sha"
        echo "    got      $stream_lib_have"
    fi
fi
if [ "$stream_lib_need_fetch" -eq 1 ]; then
    echo "Fetching NVIDIA's stream client -> web/omniverse-webrtc-streaming-library.js ..."
    # A local problem (no web/ directory, a read-only checkout, no permission) is NOT a network
    # problem - name the right one instead of sending the user off to a proxy workaround.
    if [ ! -d "$stream_lib_dir" ]; then
        echo "*** $stream_lib_dir does not exist - a LOCAL filesystem problem, not a network one. ***"
        echo "    Run the launcher from a complete checkout of the repository."
        exit 1
    fi
    if [ ! -w "$stream_lib_dir" ]; then
        echo "*** $stream_lib_dir is not writable - a LOCAL filesystem problem, not a network one. ***"
        echo "    Fix the permissions on that directory, then relaunch."
        exit 1
    fi
    # Sweep partials abandoned by killed runs (>1 day old, so never a live one).
    find "$stream_lib_dir" -maxdepth 1 -name '*.partial.*' -mtime +0 -delete 2>/dev/null || true
    rm -f "$stream_lib_tmp"   # our own leftover
    fetched=0
    fetch_rc=0
    # --max-time / --timeout bound a transfer that stalls mid-body; without them the launcher
    # hangs here indefinitely behind a lone "fetching ..." line.
    if command -v curl >/dev/null 2>&1; then
        if curl -fsSL --connect-timeout 30 --max-time 300 -o "$stream_lib_tmp" "$stream_lib_url"; then
            fetched=1
        else
            fetch_rc=$?
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget -q --timeout=30 --tries=2 -O "$stream_lib_tmp" "$stream_lib_url"; then
            fetched=1
        else
            fetch_rc=$?
        fi
    else
        echo "*** Neither curl nor wget is available to fetch it. ***"
    fi
    if [ "$fetched" -ne 1 ]; then
        rm -f "$stream_lib_tmp"   # no half-written file
        if [ "$fetch_rc" -eq 23 ]; then
            # curl exit 23 = write error: the transfer was fine, the disk was not.
            echo "*** Could not WRITE the stream client into web/ - a LOCAL filesystem problem, not a network one. ***"
            echo "    Check the permissions and the free space on: $stream_lib_dir"
        else
            echo "*** Could not download the viewer's stream client - cannot start. ***"
            echo "    On a restricted network, download it by hand from:"
            echo "      $stream_lib_url"
            echo "    and save it as:"
            echo "      $stream_lib"
        fi
        exit 1
    fi
    stream_lib_got="$(sha256_hex "$stream_lib_tmp")"
    stream_lib_rc=$?   # NOT `|| true` - that would clobber the status we are reading
    if [ "$stream_lib_rc" -eq 2 ]; then
        # The whole point of pinning a commit is integrity, so an unverifiable download does
        # not get installed - it certainly does not get announced as "SHA256 verified".
        rm -f "$stream_lib_tmp"
        echo "*** Checksum NOT verified (no sha256 tool available) - refusing to install unverified bytes. ***"
        echo "    Tried: sha256sum, shasum -a 256, openssl dgst -sha256, python3, $py"
        exit 1
    elif [ "$stream_lib_rc" -ne 0 ]; then
        # DISTINCT from a mismatch: hashing failed, so we know NOTHING about these bytes. Do not
        # call them corrupt, do not install them, and keep the download around for diagnosis.
        echo "*** Downloaded the stream client but the checksum tool failed on it - not installing it. ***"
        echo "    This is a LOCAL hashing failure, not a corrupt download."
        if [ -f "$stream_lib_tmp" ]; then
            echo "    The download was kept at: $stream_lib_tmp"
        else
            echo "    $stream_lib_tmp is not there at all - the downloader reported success but wrote nothing."
        fi
        exit 1
    elif [ "$stream_lib_got" != "$stream_lib_sha" ]; then
        # Never leave a corrupt file behind: it would half-break the viewer instead of failing.
        rm -f "$stream_lib_tmp"
        echo "*** Downloaded stream client failed its SHA256 check - file deleted, cannot start. ***"
        echo "    expected $stream_lib_sha"
        echo "    got      $stream_lib_got"
        echo "    source   $stream_lib_url"
        exit 1
    fi
    if ! mv -f "$stream_lib_tmp" "$stream_lib"; then
        # mv's exit status has to be checked: ignoring it means a failed move still prints
        # "(SHA256 verified)" and exits 0, starting the server with no library at all.
        rm -f "$stream_lib_tmp"
        echo "*** Verified the download but could not move it into place - a LOCAL filesystem problem. ***"
        echo "    $stream_lib_tmp -> $stream_lib"
        exit 1
    fi
    # Claimed only AFTER the move actually succeeded - and only on the verified path.
    echo "Fetched web/omniverse-webrtc-streaming-library.js (SHA256 verified)."
fi

child=""
on_int() {
    # Ctrl+C: stop the child too, then leave the loop for good.
    if [ -n "$child" ]; then
        kill "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    exit 130
}
trap on_int INT TERM

# Peer of the PowerShell launcher's Move-LogAside. On Linux/macOS a rename succeeds even
# while the file is an open stdout target, so the Windows "file is in use" failure does not
# normally arise here - but it DOES on a mount that carries Windows sharing semantics (a
# CIFS/SMB share, or /mnt/c under WSL), and a rename can also fail on a read-only or full
# filesystem. Rotation is a convenience, so it must never abort a launch or spam the console:
# rename, else retry briefly, else copy the content across (the fresh redirect truncates the
# original anyway, so the previous run is still preserved as *.prev.log), else note it once
# and carry on. Scoped to this one operation; nothing else in the loop is softened.
rotate_log() {
    _src="$1"
    _dst="$2"
    [ -f "$_src" ] || return 0
    _i=0
    while [ "$_i" -lt 5 ]; do
        if mv -f "$_src" "$_dst" 2>/dev/null; then return 0; fi
        sleep 0.1
        _i=$((_i + 1))
    done
    if cp -f "$_src" "$_dst" 2>/dev/null; then return 0; fi
    echo "    (note: could not rotate $(basename "$_src") - still open elsewhere; this run overwrites it)"
    return 0
}

while true; do
    rotate_log "$out" "$prev_out"
    rotate_log "$err" "$prev_err"
    rm -f "$url_file"   # stale URL from a prior launch
    started=$(date +%s)
    "$py" -u -m dev_variant_presenter --host 127.0.0.1 --port 8080 >"$out" 2>"$err" &
    child=$!
    # Surface the resolved URL (the port may have shifted off 8080). Poll for the file the
    # server drops on startup; it lands within ~1s, so this ceiling is just a safety net.
    shown_url=""
    i=0
    while [ "$i" -lt 75 ]; do
        kill -0 "$child" 2>/dev/null || break
        if [ -f "$url_file" ]; then
            shown_url="$(tr -d '\r\n' < "$url_file")"
            break
        fi
        sleep 0.2
        i=$((i + 1))
    done
    if [ -n "$shown_url" ]; then
        echo ""
        echo "*** OVRTX Dev Variant Presenter is up - open:  $shown_url ***"
        echo "    Single-client stream: use ONE tab. On a relaunch, just reload that tab."
        echo ""
    fi
    wait "$child"
    code=$?
    child=""
    up=$(( $(date +%s) - started ))
    # 0 = clean exit, 130 = SIGINT (Ctrl+C). Anything else is a crash or a self-restart
    # (/api/restart exits 43 expecting exactly this supervisor), so relaunch.
    if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
        break
    fi
    echo "$(date +%Y-%m-%dT%H:%M:%S)  exit=$code  uptime=${up}s" >> "$log"
    echo ""
    echo "*** server died (exit $code after ${up}s) - relaunching in 3s (Ctrl+C to stop) ***"
    echo ""
    sleep 3
done
