# Step A2/A3: extract ric_data.zip -> $DATA_ROOT/ric_data/<sub_id>/*.json
#
# The archive is compressed with Deflate64 (method 9). Verified NON-options:
#   - Python zipfile          -> NotImplementedError (compression type 9)
#   - .NET System.IO.Compression (PS 5.1 / .NET FW 4.x) -> "unsupported compression method"
#   - bsdtar / libarchive 3.8 -> "Damaged Zip archive"
# The Windows shell zip handler DOES support Deflate64, and needs no install.
$ErrorActionPreference = 'Stop'

$zipPath   = 'C:\Users\alvin\Overstride_V2\data\_staging\ric_data.zip'
$dataRoot  = 'C:\Users\alvin\Overstride_V2\data\ric'
$staged    = Join-Path $dataRoot 'reformat_data'   # the archive's own root folder name
$final     = Join-Path $dataRoot 'ric_data'        # what CLAUDE.md expects
$EXPECTED  = 2506

if (Test-Path $final) { throw "$final already exists - refusing to overwrite" }

$shell = New-Object -ComObject Shell.Application
$root  = $shell.NameSpace($zipPath).Items() | Select-Object -First 1
if ($root.Name -ne 'reformat_data') { throw "unexpected archive root: $($root.Name)" }

Write-Output "extracting $($root.Name) -> $dataRoot"
$sw = [Diagnostics.Stopwatch]::StartNew()

# 16 = yes-to-all, 512 = no confirm on dir create, 1024 = suppress error UI.
# The progress dialog is left enabled: suppressing it measurably slows bulk copies.
$shell.NameSpace($dataRoot).CopyHere($root, 1552)

# CopyHere is asynchronous. Poll until the JSON count stops changing at the target.
$last = -1; $stable = 0
while ($true) {
    Start-Sleep -Seconds 15
    $n = @(Get-ChildItem $staged -Recurse -Filter *.json -File -ErrorAction SilentlyContinue).Count
    $gb = [math]::Round((Get-ChildItem $staged -Recurse -File -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum / 1GB, 2)
    Write-Output ("  {0,5}/{1} json  {2} GB  {3:N0}s" -f $n, $EXPECTED, $gb, $sw.Elapsed.TotalSeconds)

    if ($n -eq $EXPECTED) { break }
    if ($n -eq $last) { $stable++ } else { $stable = 0 }
    $last = $n
    if ($stable -ge 8) { throw "stalled at $n/$EXPECTED files after $([int]$sw.Elapsed.TotalSeconds)s" }
}

$sw.Stop()
Rename-Item -Path $staged -NewName 'ric_data'
Write-Output ("DONE: {0} json files in {1:N0}s -> {2}" -f $EXPECTED, $sw.Elapsed.TotalSeconds, $final)
