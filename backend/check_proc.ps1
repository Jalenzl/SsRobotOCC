$p = Get-Process -Id 6700 -ErrorAction SilentlyContinue
if ($p) {
    Write-Output ("Id: " + $p.Id)
    Write-Output ("Name: " + $p.ProcessName)
    Write-Output ("WS_MB: " + [math]::Round($p.WorkingSet64/1MB, 1))
    Write-Output ("Path: " + $p.Path)
} else {
    Write-Output "Process 6700 not found"
}
