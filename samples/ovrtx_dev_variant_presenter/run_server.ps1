# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# OVRTX Dev Variant Presenter - server watchdog.
# The ovrtx/ovstream native layer can hard-die (no traceback), and the app can self-restart
# (POST /api/restart -> os._exit(43), e.g. the frontend's dry-pipe escalation); this relaunches
# on abnormal exit so a death costs a page reload + auto session restore, not a dead app.
# Ctrl+C exits the loop.
#
# Waits on the child process HANDLE (Start-Process -PassThru + WaitForExit), NOT a
# "| Tee-Object" pipe: an abrupt os._exit / native exit then still yields a reliable exit code.
# A pipe also wedges the watchdog whenever a parent process captures this script's stdout
# (a wrapper script, a CI job, any launcher that redirects): the pipe never closes, so the
# wait never returns - no relaunch, no crash-log line, just a powershell sitting idle.
# Stdout/stderr go to logs\ so a death still leaves evidence (split into _out / _err;
# previous run kept as *.prev.log).
# ASCII only: powershell.exe parses BOM-less .ps1 as ANSI, so non-ASCII breaks quoting.
$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $proj ".venv\Scripts\python.exe"
$log = Join-Path $proj "server_crashes.log"
$logdir = Join-Path $proj "logs"
if (-not (Test-Path $logdir)) { New-Item -ItemType Directory $logdir | Out-Null }
$out = Join-Path $logdir "server_out.log"
$err = Join-Path $logdir "server_err.log"
$prevOut = Join-Path $logdir "server_out.prev.log"
$prevErr = Join-Path $logdir "server_err.prev.log"
$urlFile = Join-Path $logdir "server_url.txt"
$env:PYTHONUNBUFFERED = "1"   # the crash tail must reach the log before the process dies
# The control port is chosen dynamically (8080 if free, else the next free port, else an
# ephemeral one). The child's stdout is redirected to a log, so it drops the resolved URL
# here for us to print to the console after each launch.
$env:DEV_VARIANT_PRESENTER_URL_FILE = $urlFile
if (-not (Test-Path $py)) {
    # First run on a fresh clone: build the environment, then fall through to the loop.
    # Kept OUT of the relaunch loop on purpose - a crash must relaunch instantly, not resync.
    Write-Host "No .venv found - running 'uv sync' (first-time setup) ..."
    Push-Location $proj
    uv sync
    $syncCode = $LASTEXITCODE
    Pop-Location
    if ($syncCode -ne 0 -or -not (Test-Path $py)) {
        Write-Host "*** uv sync failed (is uv installed? https://docs.astral.sh/uv/) - cannot start. ***"
        exit 1
    }
}
# The browser viewer needs NVIDIA's StreamSDK WebRTC client, which is NOT redistributed in
# this repo (it is NVIDIA's own software, published in NVIDIA-Omniverse/ovstream). Fetch it
# once from a PINNED upstream commit - never HEAD, so an upstream change cannot silently
# alter or break this. The Python wheels ship no JS, so uv sync cannot provide it.
# Kept OUT of the relaunch loop for the same reason as uv sync: a crash must relaunch
# instantly, not re-verify a 700 KB download.
$streamLib = Join-Path $proj "web\omniverse-webrtc-streaming-library.js"
$streamLibDir = Split-Path -Parent $streamLib
$streamLibUrl = "https://raw.githubusercontent.com/NVIDIA-Omniverse/ovstream/af7f1f9006d1037a3cc7b8eca73f39a6469b69c2/examples/webrtc_client/omniverse-webrtc-streaming-library.js"
$streamLibSha = "447a74830162b91cb92b0a636f02c0b3e668d835e2a4496f560e31e2b48e5c71"
# Download to a PER-PROCESS temp path and only move it into place once the hash checks out, so
# the real filename never exists in a half-written state: an interrupted download (Ctrl+C, a
# dropped link, a power cut) leaves the installed file untouched. The PID suffix is what keeps
# two concurrent launches out of each other's way - with a shared ".partial", one run deletes
# the other's in-flight download, or hashes its neighbour's half-written bytes. Each run writes
# its own file, and the rename into place is atomic.
$streamLibTmp = "$streamLib.partial.$PID"
# Hash WITHOUT Get-FileHash. Get-FileHash lives in Microsoft.PowerShell.Utility, and when this
# script is launched via Start-Process / CreateProcess from a PowerShell 7 parent (background
# tasks, wrapper scripts - i.e. how this project is routinely started) the child Windows
# PowerShell 5.1 inherits PS7's PSModulePath, resolves Utility to the PS7 copy, and Get-FileHash
# is then simply NOT THERE. A missing cmdlet yields $null, which would fall into the "hash
# mismatch" branch below and DELETE a byte-perfect download on every single launch.
# .NET's SHA256 needs no module at all. Returns $null on a hashing failure and puts the reason
# in $script:streamLibHashError; callers must treat that as a LOCAL failure, never as corruption.
$script:streamLibHashError = ""
function Get-Sha256Hex($path) {
    $script:streamLibHashError = ""
    $sha = $null
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $bytes = $sha.ComputeHash([System.IO.File]::ReadAllBytes($path))
        return ([System.BitConverter]::ToString($bytes)).Replace('-', '').ToLowerInvariant()
    } catch {
        $script:streamLibHashError = $_.Exception.Message
        return $null
    } finally {
        if ($sha) { $sha.Dispose() }
    }
}
# Existence is NOT integrity. A 0-byte file, a saved HTML error page, a truncated scp, a
# cloud-sync placeholder or a copy from an older pin all "exist", so a bare
# `if (-not (Test-Path ...))` gate accepts every one of them silently - a permanently broken
# viewer with no diagnostic anywhere. Re-hashing ~700 KB costs a few milliseconds, and this
# whole block stays OUT of the relaunch loop, so a crash still relaunches instantly.
$streamLibNeedFetch = $true
if (Test-Path $streamLib) {
    $streamLibHave = Get-Sha256Hex $streamLib
    if ($null -eq $streamLibHave) {
        Write-Host "*** Could not compute a SHA256 checksum for web\omniverse-webrtc-streaming-library.js. ***"
        Write-Host "    $script:streamLibHashError"
        Write-Host "    This is a LOCAL hashing failure, not a corrupt file - the file was left alone."
        Write-Host "    Release the file if something else holds it open, check its permissions, then relaunch."
        exit 1
    }
    if ($streamLibHave -eq $streamLibSha) {
        $streamLibNeedFetch = $false
    } else {
        Write-Host "web\omniverse-webrtc-streaming-library.js does not match the pinned SHA256 - re-fetching it."
        Write-Host "    expected $streamLibSha"
        Write-Host "    got      $streamLibHave"
    }
}
if ($streamLibNeedFetch) {
    Write-Host "Fetching NVIDIA's stream client -> web\omniverse-webrtc-streaming-library.js ..."
    # A local problem (no web\ directory, a read-only checkout, no permission) is NOT a network
    # problem - name the right one instead of sending the user off to a proxy workaround.
    if (-not (Test-Path $streamLibDir)) {
        Write-Host "*** $streamLibDir does not exist - a LOCAL filesystem problem, not a network one. ***"
        Write-Host "    Run the launcher from a complete checkout of the repository."
        exit 1
    }
    # Sweep partials abandoned by killed runs (>1 day old, so never a live one). Done with .NET
    # so it cannot itself depend on a module that may not have loaded.
    foreach ($stale in [System.IO.Directory]::GetFiles($streamLibDir, "*.partial.*")) {
        if ([System.IO.File]::GetLastWriteTimeUtc($stale) -lt [DateTime]::UtcNow.AddDays(-1)) {
            try { [System.IO.File]::Delete($stale) } catch { }
        }
    }
    if (Test-Path $streamLibTmp) { Remove-Item -Force -LiteralPath $streamLibTmp }   # our own leftover
    # A stalled transfer must not hang the launcher behind a lone "fetching ..." line. NOTE that
    # Invoke-WebRequest's -TimeoutSec does NOT do this: it bounds the connect + response-header
    # phase only. Measured against an origin that sends headers and then stalls mid-body,
    # -TimeoutSec 5 still took 300s to give up - that is .NET's DEFAULT ReadWriteTimeout, not our
    # setting, and a trickle slower than one read per 300s would never end at all. WebRequest
    # exposes both knobs, so use it directly. It also keeps the download off
    # Microsoft.PowerShell.Utility - the module that goes missing in the Get-FileHash case above.
    # Transfer into MEMORY first (it is 700 KB), then write. That is also what keeps the two
    # failure kinds apart: anything thrown below is a NETWORK failure, and anything thrown by the
    # write that follows is a LOCAL filesystem failure. Reporting a permission or missing-folder
    # error under a "on a restricted network, download it by hand" heading is a wrong diagnosis
    # that just wastes the user's time.
    $streamLibBytes = $null
    try {
        # PS 5.1 under an older policy can still default to TLS 1.0; harmless where not needed.
        try {
            [System.Net.ServicePointManager]::SecurityProtocol =
                [System.Net.ServicePointManager]::SecurityProtocol -bor [System.Net.SecurityProtocolType]::Tls12
        } catch { }
        $streamLibReq = [System.Net.WebRequest]::Create($streamLibUrl)
        $streamLibReq.Timeout = 60000            # connect + response headers
        $streamLibReq.ReadWriteTimeout = 60000   # ANY single stalled read of the body
        if ($streamLibReq.Proxy) { $streamLibReq.Proxy.Credentials = [System.Net.CredentialCache]::DefaultCredentials }
        $streamLibResp = $streamLibReq.GetResponse()
        try {
            $streamLibBuf = [System.IO.MemoryStream]::new()
            $streamLibIn = $streamLibResp.GetResponseStream()
            try { $streamLibIn.CopyTo($streamLibBuf) } finally { $streamLibIn.Dispose() }
            $streamLibBytes = $streamLibBuf.ToArray()
            $streamLibBuf.Dispose()
        } finally { $streamLibResp.Close() }
    } catch {
        Write-Host "*** Could not download the viewer's stream client - cannot start. ***"
        Write-Host "    $($_.Exception.Message)"
        Write-Host "    On a restricted network, download it by hand from:"
        Write-Host "      $streamLibUrl"
        Write-Host "    and save it as:"
        Write-Host "      $streamLib"
        exit 1
    }
    try {
        [System.IO.File]::WriteAllBytes($streamLibTmp, $streamLibBytes)
    } catch {
        if (Test-Path $streamLibTmp) { Remove-Item -Force -LiteralPath $streamLibTmp }   # no half-written file
        Write-Host "*** Downloaded the stream client but could not WRITE it - a LOCAL filesystem problem, not a network one. ***"
        Write-Host "    $($_.Exception.Message)"
        Write-Host "    Check the permissions and the free space on: $streamLibDir"
        exit 1
    }
    $streamLibGot = Get-Sha256Hex $streamLibTmp
    if ($null -eq $streamLibGot) {
        # DISTINCT from a mismatch: hashing failed, so we know NOTHING about these bytes. Do not
        # call them corrupt, do not install them, and keep the download around for diagnosis.
        Write-Host "*** Downloaded the stream client but could not compute its SHA256 - not installing it. ***"
        Write-Host "    $script:streamLibHashError"
        Write-Host "    This is a LOCAL hashing failure, not a corrupt download."
        if (Test-Path $streamLibTmp) {
            Write-Host "    The download was kept at: $streamLibTmp"
        } else {
            Write-Host "    $streamLibTmp is not there at all - the download reported success but wrote nothing."
        }
        exit 1
    }
    if ($streamLibGot -ne $streamLibSha) {
        # Never leave a corrupt file behind: it would half-break the viewer instead of failing.
        Remove-Item -Force -LiteralPath $streamLibTmp
        Write-Host "*** Downloaded stream client failed its SHA256 check - file deleted, cannot start. ***"
        Write-Host "    expected $streamLibSha"
        Write-Host "    got      $streamLibGot"
        Write-Host "    source   $streamLibUrl"
        exit 1
    }
    try {
        Move-Item -Force -LiteralPath $streamLibTmp -Destination $streamLib -ErrorAction Stop
    } catch {
        # The move is the last thing that can fail, so it gets its own branch: reporting
        # "(SHA256 verified)" and exiting 0 here would start the server with no library at all.
        if (Test-Path $streamLibTmp) { Remove-Item -Force -LiteralPath $streamLibTmp }
        Write-Host "*** Verified the download but could not move it into place - a LOCAL filesystem problem. ***"
        Write-Host "    $($_.Exception.Message)"
        Write-Host "    $streamLibTmp -> $streamLib"
        exit 1
    }
    # Claimed only AFTER the move actually succeeded - and only on the verified path.
    Write-Host "Fetched web\omniverse-webrtc-streaming-library.js (SHA256 verified)."
}
# Rotating a log means RENAMING a file that is a live stdout/stderr redirect target, and on
# Windows a rename is refused while any process still holds a handle to it - even though a
# fresh redirect to the same path is allowed, and reading it is allowed. Something holds it
# more often than you would think: the launcher waits on .venv\Scripts\python.exe, but that
# is a trampoline that re-execs the real interpreter, and the grandchild INHERITS these
# handles, so it can outlive the process we waited on; a previous server can still be tearing
# down (native GPU teardown is not instant) when the next launch starts; a backup or AV
# scanner can hold it for a moment. Unhandled, that surfaces as a red Move-Item error on
# every launch while the server starts fine anyway.
# Rotation is a convenience, so it must never abort a launch or spam the console: rename,
# else retry briefly (wins the sub-second teardown race), else COPY the content across (a
# read is permitted under the lock, and the fresh redirect truncates the original anyway, so
# the previous run is still preserved as *.prev.log), else note it once and carry on. The
# error handling is scoped to this one operation - nothing else in the loop is softened, and
# the fast path is untouched, so an abnormal exit still relaunches instantly.
function Move-LogAside($src, $dst) {
    if (-not (Test-Path $src)) { return }
    for ($i = 0; $i -lt 5; $i++) {
        try { Move-Item -Force -LiteralPath $src -Destination $dst -ErrorAction Stop; return } catch { }
        Start-Sleep -Milliseconds 100
    }
    try { Copy-Item -Force -LiteralPath $src -Destination $dst -ErrorAction Stop; return } catch { }
    Write-Host "    (note: could not rotate $(Split-Path -Leaf $src) - still open elsewhere; this run overwrites it)"
}
while ($true) {
    Move-LogAside $out $prevOut
    Move-LogAside $err $prevErr
    if (Test-Path $urlFile) { Remove-Item -Force $urlFile }   # stale URL from a prior launch
    $started = Get-Date
    $proc = Start-Process -FilePath $py -ArgumentList '-u', '-m', 'dev_variant_presenter', '--host', '127.0.0.1', '--port', '8080' -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $proc.Handle   # cache the handle so .ExitCode is readable after exit (.NET quirk)
    # Surface the resolved URL (the port may have shifted off 8080). Poll for the file the
    # server drops on startup; it lands within ~1s, so this ceiling is just a safety net.
    $shownUrl = $null
    for ($i = 0; $i -lt 75; $i++) {
        if ($proc.HasExited) { break }
        if (Test-Path $urlFile) { $shownUrl = (Get-Content $urlFile -Raw).Trim(); break }
        Start-Sleep -Milliseconds 200
    }
    if ($shownUrl) {
        Write-Host ""
        Write-Host "*** OVRTX Dev Variant Presenter is up - open:  $shownUrl ***"
        Write-Host "    Single-client stream: use ONE tab. On a relaunch, just reload that tab."
        Write-Host ""
    }
    $proc.WaitForExit()
    $code = $proc.ExitCode
    $up = (Get-Date) - $started
    if ($code -eq 0 -or $code -eq -1073741510) { break }   # clean exit / Ctrl+C (0xC000013A)
    Add-Content $log "$(Get-Date -Format s)  exit=$code  uptime=$([int]$up.TotalSeconds)s"
    Write-Host ""
    Write-Host "*** server died (exit $code after $([int]$up.TotalSeconds)s) - relaunching in 3s (Ctrl+C to stop) ***"
    Write-Host ""
    Start-Sleep -Seconds 3
}
