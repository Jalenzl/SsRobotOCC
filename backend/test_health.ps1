try {
    $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 5
    Write-Output $r.StatusCode
} catch {
    Write-Output ('ERR: ' + $_.Exception.Message)
}
