$p = Get-CimInstance Win32_Process -Filter "ProcessId=6700" -ErrorAction SilentlyContinue
if ($p) {
    Write-Output $p.CommandLine
} else {
    Write-Output "Not found"
}
