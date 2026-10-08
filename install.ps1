<#
.SYNOPSIS
    HiveBear installer for Windows.

.DESCRIPTION
    Installs the hivebear CLI into %LOCALAPPDATA%\HiveBear\bin and adds it to
    your user PATH. With -Desktop, also downloads and runs the desktop app's
    MSI installer.

    CLI:
        irm https://hivebear.com/install.ps1 | iex

    Desktop app (iex cannot pass parameters, so use either form):
        $env:HIVEBEAR_DESKTOP = 1; irm https://hivebear.com/install.ps1 | iex
        & ([scriptblock]::Create((irm https://hivebear.com/install.ps1))) -Desktop

    Files downloaded by PowerShell carry no Mark of the Web, so SmartScreen
    does not stop the unsigned installer. The MSI installs for all users, so
    Windows still asks for administrator approval (UAC).

.PARAMETER Desktop
    Also install the HiveBear desktop app.

.PARAMETER NoCli
    Skip the CLI (use with -Desktop to install only the desktop app).

.PARAMETER Version
    Release tag to install, e.g. v0.1.9. Defaults to the latest release, or
    $env:HIVEBEAR_VERSION if set.
#>
param(
    [switch]$Desktop,
    [switch]$NoCli,
    [string]$Version
)

# Everything lives in a function so that, when piped into iex, a failure ends
# the install rather than the user's PowerShell session (a top-level `exit`
# would close their window).
function Install-HiveBear {
    param(
        [bool]$WantDesktop,
        [bool]$WantCli,
        [string]$Tag
    )

    $ErrorActionPreference = 'Stop'
    # Invoke-WebRequest's progress bar slows downloads by an order of
    # magnitude on Windows PowerShell 5.1.
    $ProgressPreference = 'SilentlyContinue'
    # Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub refuses.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $Repo = 'BeckhamLabsLLC/HiveBear'
    $InstallRoot = Join-Path $env:LOCALAPPDATA 'HiveBear'
    $BinDir = Join-Path $InstallRoot 'bin'

    Write-Host 'HiveBear Installer'
    Write-Host '=================='
    Write-Host ''

    $arch = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) { $arch = $env:PROCESSOR_ARCHITEW6432 }
    switch ($arch) {
        'AMD64' { }
        'ARM64' {
            Write-Host 'Note: there is no native ARM64 build yet. Installing the x64 build,'
            Write-Host 'which runs under Windows 11 x64 emulation.'
        }
        default {
            throw "Unsupported architecture: $arch. HiveBear needs 64-bit Windows."
        }
    }

    if (-not $Tag) { $Tag = 'latest' }
    if ($Tag -eq 'latest') {
        $BaseUrl = "https://github.com/$Repo/releases/latest/download"
    } else {
        if (-not $Tag.StartsWith('v')) { $Tag = "v$Tag" }
        $BaseUrl = "https://github.com/$Repo/releases/download/$Tag"
    }

    Write-Host "Version:   $Tag"
    Write-Host ''

    $Work = Join-Path ([IO.Path]::GetTempPath()) ("hivebear-install-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $Work -Force | Out-Null

    try {
        if ($WantCli) {
            Install-Cli -BaseUrl $BaseUrl -Work $Work -BinDir $BinDir -Repo $Repo
        }
        if ($WantDesktop) {
            if ($WantCli) { Write-Host '' }
            Install-Desktop -BaseUrl $BaseUrl -Work $Work -Repo $Repo
        }
    } finally {
        Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
    }
}

function Get-File {
    param([string]$Url, [string]$OutFile)
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
    } catch {
        throw "Download failed: $Url ($($_.Exception.Message))"
    }
}

# Same rule as install.sh: an exact match on the filename column of a
# sha256sum-format file ("<hash>  <name>" or "<hash> *<name>"), and refuse to
# install anything the release does not list.
function Test-Checksum {
    param([string]$File, [string]$SumsFile, [string]$Name, [string]$Repo)

    $expected = $null
    foreach ($line in Get-Content -LiteralPath $SumsFile) {
        $parts = $line.Trim() -split '\s+', 2
        if ($parts.Count -eq 2 -and ($parts[1] -eq $Name -or $parts[1] -eq "*$Name")) {
            $expected = $parts[0].ToLowerInvariant()
            break
        }
    }
    if (-not $expected) {
        throw "No checksum for $Name in $(Split-Path -Leaf $SumsFile). Aborting."
    }

    $actual = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw ("Checksum verification failed for $Name.`n  Expected: $expected`n  Actual:   $actual`n" +
            "The download may be corrupted or tampered with. Try again, or download manually from https://github.com/$Repo/releases/latest")
    }
    Write-Host 'Checksum OK'
}

function Install-Cli {
    param([string]$BaseUrl, [string]$Work, [string]$BinDir, [string]$Repo)

    $name = 'hivebear-x86_64-pc-windows-msvc.zip'
    Write-Host 'Installing the HiveBear CLI'
    Write-Host "Downloading $name..."
    $zip = Join-Path $Work $name
    Get-File "$BaseUrl/$name" $zip

    Write-Host 'Verifying checksum...'
    $sums = Join-Path $Work 'SHA256SUMS.txt'
    Get-File "$BaseUrl/SHA256SUMS.txt" $sums
    Test-Checksum -File $zip -SumsFile $sums -Name $name -Repo $Repo

    $extract = Join-Path $Work 'cli'
    Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
    $exe = Get-ChildItem -LiteralPath $extract -Recurse -Filter 'hivebear.exe' | Select-Object -First 1
    if (-not $exe) { throw "$name did not contain hivebear.exe" }

    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    $dest = Join-Path $BinDir 'hivebear.exe'
    try {
        Copy-Item -LiteralPath $exe.FullName -Destination $dest -Force
    } catch {
        throw "Could not write $dest. If hivebear is running (for example 'hivebear serve'), stop it and try again."
    }
    Write-Host "Installed hivebear to $dest"

    # User PATH, read from the registry rather than $env:Path so the machine
    # PATH is not copied into it.
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @()
    if ($userPath) { $entries = $userPath -split ';' | Where-Object { $_ } }
    if ($entries -notcontains $BinDir) {
        [Environment]::SetEnvironmentVariable('Path', (($entries + $BinDir) -join ';'), 'User')
        Write-Host "Added $BinDir to your user PATH. New terminals will pick it up."
    }
    if (($env:Path -split ';') -notcontains $BinDir) {
        $env:Path = "$env:Path;$BinDir"
    }

    $ran = $false
    try {
        $v = & $dest --version 2>&1
        if ($LASTEXITCODE -eq 0) { $ran = $true; Write-Host "Verified: $v" }
    } catch {
        $v = $_.Exception.Message
    }
    if (-not $ran) {
        Write-Host "Warning: the installed binary did not run: $v"
        Write-Host 'If Windows reports a missing VCRUNTIME140.dll, install the Microsoft Visual C++ Redistributable:'
        Write-Host '  https://aka.ms/vs/17/release/vc_redist.x64.exe'
    }

    Write-Host ''
    Write-Host 'Get started by running:'
    Write-Host ''
    Write-Host '  hivebear quickstart'
    Write-Host ''
}

function Install-Desktop {
    param([string]$BaseUrl, [string]$Work, [string]$Repo)

    $name = 'HiveBear-x64.msi'
    Write-Host 'Installing the HiveBear desktop app'
    Write-Host "Downloading $name..."
    $msi = Join-Path $Work $name
    Get-File "$BaseUrl/$name" $msi

    Write-Host 'Verifying checksum...'
    $sums = Join-Path $Work 'SHA256SUMS-desktop.txt'
    Get-File "$BaseUrl/SHA256SUMS-desktop.txt" $sums
    Test-Checksum -File $msi -SumsFile $sums -Name $name -Repo $Repo

    Write-Host 'Running the installer. Windows will ask for administrator approval.'
    $log = Join-Path $Work 'msi.log'
    $p = Start-Process -FilePath 'msiexec.exe' -ArgumentList @('/i', "`"$msi`"", '/passive', '/norestart', '/l*v', "`"$log`"") -Wait -PassThru
    switch ($p.ExitCode) {
        0 { Write-Host 'Installed. Launch HiveBear from the Start menu.' }
        3010 { Write-Host 'Installed. Restart Windows to finish.' }
        1602 { throw 'Installation was cancelled.' }
        default {
            $kept = Join-Path ([IO.Path]::GetTempPath()) 'hivebear-msi.log'
            Copy-Item -LiteralPath $log -Destination $kept -Force -ErrorAction SilentlyContinue
            throw "The installer failed (msiexec exit code $($p.ExitCode)). Log: $kept"
        }
    }
    Write-Host ''
}

$wantDesktop = $Desktop.IsPresent -or ($env:HIVEBEAR_DESKTOP -and $env:HIVEBEAR_DESKTOP -ne '0')
$wantCli = -not $NoCli.IsPresent -and -not ($env:HIVEBEAR_NO_CLI -and $env:HIVEBEAR_NO_CLI -ne '0')
if (-not $wantCli -and -not $wantDesktop) { $wantCli = $true }
$tag = $Version
if (-not $tag) { $tag = $env:HIVEBEAR_VERSION }

Install-HiveBear -WantDesktop $wantDesktop -WantCli $wantCli -Tag $tag
