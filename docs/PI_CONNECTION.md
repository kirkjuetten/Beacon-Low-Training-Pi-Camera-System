# Raspberry Pi Connection Runbook (Beginner-Friendly)

This guide is written for users with little or no terminal/SSH experience.

## What You Need Before Starting

1. Your Windows laptop is on the same network as the Pi.
2. You have a local PowerShell terminal open on Windows.
3. You have authorized credentials for `beaconpi` access.

## Important: Where To Type Commands

Type these commands in a Windows PowerShell terminal on your PC.

- OK: VS Code integrated PowerShell terminal.
- OK: Standalone Windows PowerShell.
- Not OK: A terminal that is already connected to another machine.

## Standard Connection (Most Common)

1. In Windows PowerShell, run:

```powershell
ssh pi@pi-inspect
```

2. If prompted for password, type it and press Enter.

3. Success looks like this prompt:

```text
pi@pi-inspect:~ $
```

That prompt means you are now logged into the Pi.

## First-Time Prompt You May See (Normal)

You may see:

```text
The authenticity of host 'pi-inspect (...)' can't be established.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

What to do:

1. Type `yes` and press Enter.
2. Then enter the password when asked.

You should only need this trust step once per machine/user unless host keys change.

## Quick Health Check Before SSH (If Connection Fails)

Run these on Windows PowerShell:

```powershell
Test-Connection pi-inspect -Count 1
Test-NetConnection pi-inspect -Port 22
```

Expected:

1. Ping responds.
2. `TcpTestSucceeded : True` for port 22.

## If Hostname Fails, Use IP Directly

If `ssh pi@pi-inspect` fails due to name resolution, run:

```powershell
ssh pi@192.168.226.92
```

## If You See a Host Key Mismatch Warning

This can happen after Pi reimage/rebuild. On Windows PowerShell run:

```powershell
ssh-keygen -R pi-inspect
ssh-keygen -R 192.168.226.92
ssh pi@pi-inspect
```

Then accept the authenticity prompt again by typing `yes`.

## If You Accidentally Ran `ssh-keygen -R` Earlier

That is safe. It only removes saved trust entries.

- You will be asked the authenticity question again.
- Type `yes` once to re-trust the Pi.
- This is expected behavior.

## Confirm You Are On The Pi

After login, run:

```bash
hostname
pwd
```

Typical output includes hostname `pi-inspect`.

## Remote-SSH vs Plain Terminal (Simple Rule)

1. Use plain SSH terminal for quick checks and running commands.
2. Use VS Code Remote-SSH if you want full remote file editing in VS Code.

## Access Authorization

Do not store the Beacon Pi password in this repository.

For authorized access to `beaconpi` credentials, contact:

1. Kirk Juetten
2. James Cowdery

## Daily Fast Path (Use This Next Time)

Use this exact sequence at the start of a work session.

1. Connect to Pi:

```powershell
ssh -4 pi@pi-inspect
```

2. In the Pi shell, update code:

```bash
cd ~/Beacon-Low-Training-Pi-Camera-System
./scripts/pi-update-branch.sh <branch-name>
```

3. Run a quick smoke test:

```bash
python3 -m inspection_system.app.capture_test pilot-readiness
python3 -m inspection_system.app.capture_test list-projects
python3 -m inspection_system.app.capture_test capture
python3 -m inspection_system.app.capture_test inspect
```

If those three commands run, the system is in a good state for testing/debugging.
If those four commands run, the system is in a good state for testing/debugging.

## How We Work (Repeatable Dev Loop)

This is the team workflow we used successfully:

1. Edit code locally in VS Code.
2. Commit and push to GitHub branch.
3. On Pi over SSH: `./scripts/pi-update-branch.sh <branch-name>`.
4. Run target command (`capture`, `inspect`, or other).
5. Observe behavior on real hardware and report output.
6. Repeat.

This loop is expected and is the recommended way to develop this project.

## SSH vs Desktop/VNC (What You Can Expect)

### Works over SSH

1. Pull code updates.
2. Run headless commands like `capture` and `inspect`.
3. Read pass/fail and debug output paths.

### Requires Pi Desktop or VNC

1. Dashboard GUI.
2. Interactive GUI-based training/project windows.

If you run dashboard from plain SSH, you should see a message saying a graphical desktop session is required. That is expected behavior.

## One-Time Setup Items Already Completed

These were the main blockers this session and are now fixed:

1. Missing Python dependencies on Pi (`python3-skimage`, `python3-sklearn`) installed.
2. `capture_test.py` import fix for `IndicatorLED` merged.
3. Headless dashboard behavior improved to show clear guidance instead of traceback.

You should not need to redo these each session.

## Will It Be This Difficult Every Time?

No. Today included one-time setup and bug-fix work. Typical next session should be:

1. SSH in.
2. `./scripts/pi-update-branch.sh <branch-name>`.
3. Run smoke commands.

If anything fails, capture terminal output and continue the normal debug loop.

## UX Hardening Stages (Current Status)

The staged navigation improvements are now applied in order:

1. Stage 1 complete: safer launch paths in Project Manager and explicit HOME action in training UI.
2. Stage 2 complete: launching Project Manager from Dashboard now closes Dashboard to keep one primary operator window.
3. Stage 3 complete: regression coverage added for launch close policy and navigation behavior checks.

Operator expectation after Stage 2:

1. Open Project Manager from Dashboard.
2. Dashboard closes automatically.
3. Use `Back to Dashboard` inside Project Manager to return.

## Touchscreen Keyboard Support

Tk text fields now trigger an on-screen keyboard automatically in Dashboard and Project Manager.

Supported keyboard apps (auto-detected in this order):

1. `wvkbd-mobintl`
2. `wvkbd`
3. `matchbox-keyboard`
4. `onboard`

If no keyboard appears on Pi, install one package and retry:

```bash
sudo apt-get update
sudo apt-get install -y matchbox-keyboard
```

After install, restart the app (`dashboard` or `project-manager`) and tap any text input field.

## One-Tap Desktop Shortcut (Recommended)

To make startup easy and stable across updates, use the included launcher installer once on the Pi desktop session.

Run this one time on Pi:

```bash
cd ~/Beacon-Low-Training-Pi-Camera-System
chmod +x scripts/pi-launch-dashboard.sh scripts/install_pi_desktop_launcher.sh
./scripts/install_pi_desktop_launcher.sh
```

What it creates:

1. App launcher: `~/.local/share/applications/beacon-inspection-dashboard.desktop`
2. Desktop icon: `~/Desktop/Beacon Inspection Dashboard.desktop`

Daily use:

1. Pull updates as normal: `git pull origin feature/next-phase-ui-workflow`
1. Pull updates as normal: `./scripts/pi-update-branch.sh <branch-name>`
2. Tap **Beacon Inspection Dashboard** desktop icon.

Why this stays stable:

1. The shortcut always calls `scripts/pi-launch-dashboard.sh` in your repo path.
2. As code updates, the launcher path stays the same.
3. Simple logs are written to `~/beacon-dashboard.log` for troubleshooting.
