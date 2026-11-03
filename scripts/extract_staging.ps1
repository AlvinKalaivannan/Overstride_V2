# Step A1: pull the two nested archives out of the outer zip.
# The outer zip is STORED (compressed == uncompressed), so this is a byte copy.
# -Src is required because the download location is the caller's,
# not the repository's. -Staging defaults inside the repo. These
# were absolute paths under a developer's home directory.
param(
  [Parameter(Mandatory=$true)][string]$Src,
  [string]$Staging
)

Add-Type -AssemblyName System.IO.Compression.FileSystem

$repo = Split-Path -Parent $PSScriptRoot
if (-not $Staging) { $Staging = Join-Path $repo 'data\_staging' }
$ErrorActionPreference = 'Stop'
$src     = $Src
$staging = $Staging
if (-not (Test-Path $staging)) { New-Item -ItemType Directory -Path $staging | Out-Null }

$zip = [System.IO.Compression.ZipFile]::OpenRead($src)
try {
    foreach ($name in 'Supplemental_materials.zip', 'ric_data.zip') {
        $entry = $zip.GetEntry($name)
        $dest  = Join-Path $staging $name
        $sw    = [Diagnostics.Stopwatch]::StartNew()
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
        $sw.Stop()
        $gb = [math]::Round((Get-Item $dest).Length / 1GB, 2)
        Write-Output ("{0}: {1} GB in {2:N1}s" -f $name, $gb, $sw.Elapsed.TotalSeconds)
        if ((Get-Item $dest).Length -ne $entry.Length) { throw "size mismatch on $name" }
    }
} finally { $zip.Dispose() }
Write-Output 'STAGING EXTRACT COMPLETE'
