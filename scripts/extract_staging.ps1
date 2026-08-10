# Step A1: pull the two nested archives out of the outer zip.
# The outer zip is STORED (compressed == uncompressed), so this is a byte copy.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$src     = 'C:\Users\alvin\Downloads\24255795.zip'
$staging = 'C:\Users\alvin\Overstride_V2\data\_staging'
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
