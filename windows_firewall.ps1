# Stops the "allow Slippi Dolphin on public/private networks?" prompts (asks for admin once).
#
#   powershell -ExecutionPolicy Bypass -File windows_firewall.ps1                          # ports 51441-51700
#   powershell -ExecutionPolicy Bypass -File windows_firewall.ps1 -BasePort 52000 -Count 64
#
# Every Dolphin listens on its Slippi port, and on Windows each env runs its own copy of
# Dolphin (dolphin\instances\slip<port>, see smash_rl/env.py), so Windows asks once per
# new port. libmelee only talks to Dolphin over localhost, which the firewall doesn't
# filter, so these rules simply block inbound network traffic to every copy. Rules can
# name copies that don't exist yet. Undo with:
#   Remove-NetFirewallRule -Group "Smash RL Dolphin"   (as admin)
param([int]$BasePort = 51441, [int]$Count = 260)

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    try {
        $proc = Start-Process powershell -Verb RunAs -Wait -PassThru -ErrorAction Stop -ArgumentList (
            "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -BasePort $BasePort -Count $Count")
    } catch {
        Write-Error "Needs admin rights, and the prompt was declined: no rules were added."
        exit 1
    }
    exit $proc.ExitCode
}
$ErrorActionPreference = "Stop"

$dolphin = Join-Path $PSScriptRoot "dolphin"
$group = "Smash RL Dolphin"
Remove-NetFirewallRule -Group $group -ErrorAction SilentlyContinue
# Rules Windows made when a prompt was answered for one of these copies.
Get-NetFirewallApplicationFilter | Where-Object { $_.Program -like "$dolphin\*" } |
    Get-NetFirewallRule | Remove-NetFirewallRule

$programs = @(Join-Path $dolphin "Slippi Dolphin\Slippi Dolphin.exe")
foreach ($port in $BasePort..($BasePort + $Count - 1)) {
    $programs += Join-Path $dolphin ("instances\slip{0:D5}\Slippi Dolphin.exe" -f $port)
}
foreach ($program in $programs) {
    $name = Split-Path (Split-Path $program -Parent) -Leaf
    New-NetFirewallRule -DisplayName "$group ($name)" -Group $group -Direction Inbound -Action Block `
        -Program $program -Profile Any | Out-Null
}
Write-Host "Blocked inbound network access for $($programs.Count) Dolphin copies (ports $BasePort-$($BasePort + $Count - 1))."
