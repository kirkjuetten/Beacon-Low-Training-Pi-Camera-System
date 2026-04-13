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
