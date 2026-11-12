param(
    [Parameter(Mandatory = $true)]
    [string]$JobPath,
    [string]$LibraryDirectory = 'E:\FragPipe\FragPipe-24.0\tools\MSFragger-4.4.1\ext\thermo'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$InformationPreference = 'SilentlyContinue'
$WarningPreference = 'SilentlyContinue'
$VerbosePreference = 'SilentlyContinue'
$DebugPreference = 'SilentlyContinue'

$writer = $null
$exitCode = 0
$currentRawIndex = $null
$currentScan = $null

try {
    $job = Get-Content -LiteralPath $JobPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Add-Type -Path (Join-Path $LibraryDirectory 'ThermoFisher.CommonCore.Data.dll') | Out-Null
    Add-Type -Path (Join-Path $LibraryDirectory 'ThermoFisher.CommonCore.RawFileReader.dll') | Out-Null

    # stdout is a binary protocol. Progress and errors go exclusively to stderr.
    $writer = [System.IO.BinaryWriter]::new([Console]::OpenStandardOutput())
    $writer.Write([System.Text.Encoding]::ASCII.GetBytes('NMS2C001'))

    foreach ($file in @($job.files)) {
        $currentRawIndex = [int]$file.raw_index
        $currentScan = $null
        if ($currentRawIndex -lt 0) { throw 'raw_index must be nonnegative.' }
        $rawPath = [string]$file.path
        $scanNumbers = @($file.scans)
        $before = Get-Item -LiteralPath $rawPath
        $beforeLength = $before.Length
        $beforeWriteTime = $before.LastWriteTimeUtc
        $raw = $null
        $completedScans = 0
        $methodCounts = [int[]]::new(3)

        try {
            $raw = [ThermoFisher.CommonCore.RawFileReader.RawFileReaderAdapter]::FileFactory($rawPath)
            if (-not $raw.IsOpen -or $raw.IsError -or $raw.FileError.HasError) {
                throw "Cannot open RAW: $rawPath; $($raw.FileError.ErrorMessage)"
            }
            [void]$raw.SelectInstrument([ThermoFisher.CommonCore.Data.Business.Device]::MS, 1)

            foreach ($scanNumber in $scanNumbers) {
                $currentScan = [int]$scanNumber
                $filter = $raw.GetFilterForScanNumber($currentScan)
                if ([int]$filter.MSOrder -ne 2) {
                    throw "Requested scan $currentScan in $rawPath is not MS2."
                }
                $stats = $raw.GetScanStatsForScanNumber($currentScan)
                [double[]]$mz = $null
                [double[]]$intensity = $null
                [byte]$method = 0

                if ($stats.IsCentroidScan) {
                    # Preserve the complete analysis spectrum already stored in the RAW.
                    $spectrum = $raw.GetSegmentedScanFromScanNumber($currentScan, $stats)
                    $mz = $spectrum.Positions
                    $intensity = $spectrum.Intensities
                }
                else {
                    # false excludes separately flagged reference/exception peaks.
                    $centroidStream = $raw.GetCentroidStream($currentScan, $false)
                    if ($null -ne $centroidStream -and $centroidStream.Length -gt 0) {
                        $mz = $centroidStream.Masses
                        $intensity = $centroidStream.Intensities
                        $method = 1
                    }
                    else {
                        $scan = [ThermoFisher.CommonCore.Data.Business.Scan]::FromFile($raw, $currentScan)
                        $centroid = [ThermoFisher.CommonCore.Data.Business.Scan]::ToCentroid($scan)
                        $mz = $centroid.PreferredMasses
                        $intensity = $centroid.PreferredIntensities
                        $method = 2
                    }
                }

                if ($null -eq $mz -or $null -eq $intensity) {
                    throw "Null peak arrays at scan $currentScan in $rawPath."
                }
                if ($mz.Length -ne $intensity.Length) {
                    throw "Peak array length mismatch at scan $currentScan in $rawPath."
                }
                [int]$count = $mz.Length
                [long]$byteCount = [long]$count * 8
                if ($byteCount -gt [int]::MaxValue) { throw 'Peak array exceeds the binary record limit.' }
                $reaction = $filter.GetReaction(0)

                # Fixed 29-byte little-endian header: i32, i32, f64, f64, i32, u8.
                $writer.Write([int]$currentRawIndex)
                $writer.Write([int]$currentScan)
                $writer.Write([double]$reaction.PrecursorMass)
                $writer.Write([double]($stats.StartTime * 60.0))
                $writer.Write([int]$count)
                $writer.Write([byte]$method)

                # No intensity scaling, peak filtering, sorting, or padding.
                $peakBytes = [byte[]]::new([int]$byteCount)
                [System.Buffer]::BlockCopy($mz, 0, $peakBytes, 0, [int]$byteCount)
                $writer.Write($peakBytes)
                [System.Buffer]::BlockCopy($intensity, 0, $peakBytes, 0, [int]$byteCount)
                $writer.Write($peakBytes)
                $completedScans++
                $methodCounts[$method]++
            }
        }
        finally {
            if ($null -ne $raw) { $raw.Dispose() }
            $after = Get-Item -LiteralPath $rawPath
            if ($after.Length -ne $beforeLength -or $after.LastWriteTimeUtc -ne $beforeWriteTime) {
                throw "Source RAW size or modification time changed while reading: $rawPath"
            }
        }

        $writer.Flush()
        $progressJson = [ordered]@{
            type = 'raw_complete'
            raw_index = $currentRawIndex
            scan_count = $completedScans
            native_centroid = $methodCounts[0]
            stored_centroid_stream = $methodCounts[1]
            converted_centroid = $methodCounts[2]
            source_unchanged = $true
        } | ConvertTo-Json -Compress
        [Console]::Error.WriteLine($progressJson)
    }

    # A four-byte negative raw_index terminates the stream; no header follows it.
    $writer.Write([int]-1)
    $writer.Flush()
}
catch {
    $exitCode = 1
    $errorJson = [ordered]@{
        type = 'error'
        raw_index = $currentRawIndex
        scan_number = $currentScan
        message = $_.Exception.Message
    } | ConvertTo-Json -Compress
    [Console]::Error.WriteLine($errorJson)
}
finally {
    if ($null -ne $writer) {
        try { $writer.Dispose() }
        catch {
            $exitCode = 1
            [Console]::Error.WriteLine('Failed to close the binary output stream: ' + $_.Exception.Message)
        }
    }
}

exit $exitCode
